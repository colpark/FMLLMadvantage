"""Tests for the verifier sources and integrator.

Coverage:
    - Verdict / claim / config schemas validate.
    - Each source returns the right SourceVerdict on hand-crafted
      pass/fail bridged objects.
    - The integrator aggregates across sources correctly.
    - SourcesConfig ablation (V0..V4) propagates as SKIP for
      disabled sources, matching the architectural commitment from
      the addendum.
    - Literature lookup with the actual shipped clusters.json.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fmllm.bridges import (
    FMContext,
    make_structure_bridge,
)
from fmllm.bridges.compose import metadata_yaml_path
from fmllm.fms._calibration import compute_cross_fm_tolerances
from fmllm.fms._schemas import (
    BridgedFMOutput,
    ProbeReport,
    ProbeResult,
    load_fm_metadata,
)
from fmllm.fms._schemas.probe_schema import now_utc_iso
from fmllm.verifier import (
    ConformalSource,
    CrossFMSource,
    LiteratureSource,
    PhysicalStateClaim,
    RuleLibrarySource,
    SimulatorSource,
    SourceDecision,
    SourcesConfig,
    Verifier,
    VerifierVerdict,
    build_default_verifier,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LITERATURE_DB = REPO_ROOT / "data" / "literature" / "clusters.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_context(fm_name: str, *, with_calibration: bool = True) -> FMContext:
    metadata = load_fm_metadata(metadata_yaml_path(fm_name))
    probe_results = [
        ProbeResult(
            constraint_name=c.name,
            satisfaction_score=0.85,
            num_test_cases=64,
            metric="synthetic",
            passes_threshold=True,
            threshold=c.expected_satisfaction,
            details={},
        )
        for c in metadata.physics_constraints
    ]
    probe_report = ProbeReport(
        fm_name=metadata.name, fm_version=metadata.version,
        timestamp_utc=now_utc_iso(), results=probe_results,
    )
    calibration: dict = {}
    if with_calibration:
        thresholds = {
            "fm1_image": {"0.1000": 0.76, "0.2000": 0.49},
            "fm2_rdf": {"0.1000": 0.07, "0.2000": 0.045},
            "fm3_traj": {"0.1000": 1.19, "0.2000": 0.88},
        }[fm_name]
        calibration = {
            "fm_name": fm_name, "score_name": "synthetic",
            "thresholds": thresholds, "extra": {},
        }
    return FMContext(
        fm_name=fm_name, metadata=metadata,
        probe_report=probe_report, calibration=calibration,
    )


def _bridged_fm1(positions=None, n_pred=7, confidences=None) -> BridgedFMOutput:
    ctx = _build_context("fm1_image")
    if positions is None:
        positions = [[0.5, 0.3], [-1.2, 0.7], [0.0, -1.5]]
    if confidences is None:
        confidences = [3.0] * len(positions)
    raw = {
        "count_logits": torch.cat([
            torch.full((30,), -3.0),
            torch.tensor([5.0]),
        ]),
        "positions": torch.tensor(positions),
        "confidence_logits": torch.tensor(confidences),
    }
    # Override the count argmax to land at n_pred.
    raw["count_logits"] = torch.full((31,), -3.0)
    raw["count_logits"][n_pred] = 5.0
    return make_structure_bridge(ctx).emit(raw, input_provenance={"specimen_id": 0})


def _bridged_fm2(energy: float = -1.42) -> BridgedFMOutput:
    ctx = _build_context("fm2_rdf")
    return make_structure_bridge(ctx).emit(
        {"energy": torch.tensor(energy)},
        input_provenance={"specimen_id": 0},
    )


def _bridged_fm3(alpha: float = 2.0, beta: float = 0.55) -> BridgedFMOutput:
    ctx = _build_context("fm3_traj")
    return make_structure_bridge(ctx).emit(
        {"alpha": torch.tensor(alpha), "beta": torch.tensor(beta)},
        input_provenance={"specimen_id": 0},
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_sources_config_ablation_presets():
    assert SourcesConfig.for_ablation("V0").active_sources() == []
    assert "rule_library" in SourcesConfig.for_ablation("V1").active_sources()
    assert SourcesConfig.for_ablation("V4").active_sources() == [
        "rule_library", "literature", "cross_fm", "simulator", "conformal",
    ]


def test_sources_config_unknown_level_rejected():
    with pytest.raises(ValueError):
        SourcesConfig.for_ablation("V99")


# ---------------------------------------------------------------------------
# Rule library source
# ---------------------------------------------------------------------------


def test_rule_library_passes_clean_inputs():
    src = RuleLibrarySource()
    bridged = [_bridged_fm2(energy=-1.42), _bridged_fm3()]
    claim = PhysicalStateClaim(n_atoms=7)
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.PASS


def test_rule_library_flags_non_negativity_violation():
    """Energy below the LJ floor produces a hard FAIL."""
    src = RuleLibrarySource()
    bridged = [_bridged_fm2(energy=-50.0)]
    verdict = src.check(bridged, PhysicalStateClaim())
    assert verdict.decision == SourceDecision.FAIL


def test_rule_library_caveat_on_position_outside_box():
    """positions_in_box was softened from hard to soft after Phase 8a.
    FM1's regression head leaks 1-5 of 20-30 atoms outside the box on
    dense clusters, which would otherwise reject correct claims. The
    constraint still fires, but as a CAVEAT not a FAIL."""
    src = RuleLibrarySource()
    bridged = [
        _bridged_fm1(
            positions=[[0.0, 0.0], [10.0, 0.0]],
            confidences=[3.0, 3.0],
            n_pred=2,
        ),
    ]
    verdict = src.check(bridged, PhysicalStateClaim())
    assert verdict.decision == SourceDecision.CAVEAT


def test_rule_library_caveat_on_atom_count_inconsistency():
    src = RuleLibrarySource()
    # n_pred=7 but only 3 confident queries. atom_count_consistency is
    # declared 'soft' in FM1 metadata, so failure produces CAVEAT
    # rather than aggregate FAIL.
    bridged = [_bridged_fm1(n_pred=7)]
    verdict = src.check(bridged, PhysicalStateClaim())
    assert verdict.decision == SourceDecision.CAVEAT


# ---------------------------------------------------------------------------
# Cross-FM source
# ---------------------------------------------------------------------------


def test_cross_fm_skips_when_no_shared_variables():
    src = CrossFMSource()
    verdict = src.check([_bridged_fm2()], PhysicalStateClaim())
    assert verdict.decision == SourceDecision.SKIP


def test_cross_fm_pass_when_claim_matches_fm1_count():
    src = CrossFMSource()
    bridged = [_bridged_fm1(n_pred=7, positions=[[0, 0]] * 7,
                            confidences=[3.0] * 7)]
    claim = PhysicalStateClaim(n_atoms=7)
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.PASS


def test_cross_fm_caveat_when_claim_disagrees_with_fm1_count():
    src = CrossFMSource()
    bridged = [_bridged_fm1(n_pred=7, positions=[[0, 0]] * 7,
                            confidences=[3.0] * 7)]
    claim = PhysicalStateClaim(n_atoms=9)
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.CAVEAT


def test_cross_fm_with_calibrated_tolerance():
    """When a tolerance matrix is supplied, the calibrated threshold
    overrides the default."""
    matrix = compute_cross_fm_tolerances(
        records=[
            {"atom_count": {"fm1_image": float(i), "claim": float(i)}}
            for i in (5, 7, 9, 11, 13, 17, 19, 21, 25, 30)
        ],
        alpha_levels=(0.10, 0.20),
        train_split="train_50k",
    )
    src = CrossFMSource(tolerance_matrix=matrix)
    bridged = [_bridged_fm1(n_pred=7, positions=[[0, 0]] * 7,
                            confidences=[3.0] * 7)]
    claim = PhysicalStateClaim(n_atoms=7)
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.PASS


# ---------------------------------------------------------------------------
# Conformal source
# ---------------------------------------------------------------------------


def test_conformal_passes_when_in_distribution_and_claim_in_band():
    src = ConformalSource()
    bridged = [_bridged_fm2(energy=-1.42)]
    claim = PhysicalStateClaim(per_atom_potential_energy=-1.40)
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.PASS


def test_conformal_caveat_when_claim_outside_band():
    src = ConformalSource()
    bridged = [_bridged_fm2(energy=-1.42)]
    # FM2 calibration band: -1.42 +/- 0.07 -> [-1.49, -1.35].
    claim = PhysicalStateClaim(per_atom_potential_energy=-2.0)
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.CAVEAT


def test_conformal_caveat_when_fm_flags_ood():
    src = ConformalSource()
    ctx = _build_context("fm2_rdf")
    bridge = make_structure_bridge(ctx)
    bridged = [bridge.emit({"energy": torch.tensor(-1.0)}, in_distribution=False)]
    verdict = src.check(bridged, PhysicalStateClaim())
    assert verdict.decision == SourceDecision.CAVEAT


# ---------------------------------------------------------------------------
# Simulator source
# ---------------------------------------------------------------------------


def test_simulator_skips_without_positions():
    src = SimulatorSource(n_steps=10)
    verdict = src.check([_bridged_fm3()], PhysicalStateClaim(temperature=0.5))
    assert verdict.decision == SourceDecision.SKIP


def test_simulator_passes_for_consistent_claim():
    """A claim with positions at the LJ minimum and a low temperature
    runs cleanly under MD without divergence."""
    from fmllm.physics import equilibrium_positions

    pos = equilibrium_positions(7, motif="triangular_disk").tolist()
    src = SimulatorSource(n_steps=20, dt=0.005, max_radius=20.0)
    claim = PhysicalStateClaim(
        n_atoms=7, motif="triangular_disk",
        temperature=0.2, positions=pos,
    )
    verdict = src.check([], claim)
    assert verdict.decision in (SourceDecision.PASS, SourceDecision.CAVEAT)
    assert "observed_T" in verdict.evidence


# ---------------------------------------------------------------------------
# Literature source
# ---------------------------------------------------------------------------


def test_literature_default_passes_on_match_regardless_of_energy():
    """Default mode (compare_energy=False) PASSes once (N, motif) match,
    even when the candidate energy differs from the ground-state
    reference. This is the post-Phase-8a default: ground-state
    references should not flag finite-T data."""
    src = LiteratureSource(LITERATURE_DB)
    # Energy is far from the ground-state reference but the cluster
    # type matches. Default mode should not CAVEAT.
    bridged = [_bridged_fm2(energy=0.5)]
    claim = PhysicalStateClaim(n_atoms=7, motif="triangular_disk")
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.PASS
    assert verdict.evidence["compare_energy"] is False


def test_literature_compare_energy_passes_for_canonical_cluster():
    """With compare_energy=True an FM2 prediction near the literature
    reference still passes."""
    src = LiteratureSource(
        LITERATURE_DB, compare_energy=True, energy_tolerance=0.30,
    )
    bridged = [_bridged_fm2(energy=-1.79)]  # close to N=7 triangular_disk
    claim = PhysicalStateClaim(n_atoms=7, motif="triangular_disk")
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.PASS


def test_literature_compare_energy_caveat_for_strong_disagreement():
    """With compare_energy=True, a far-off energy still triggers CAVEAT."""
    src = LiteratureSource(
        LITERATURE_DB, compare_energy=True, energy_tolerance=0.10,
    )
    bridged = [_bridged_fm2(energy=0.5)]  # nowhere near the LJ minimum
    claim = PhysicalStateClaim(n_atoms=7, motif="triangular_disk")
    verdict = src.check(bridged, claim)
    assert verdict.decision == SourceDecision.CAVEAT


def test_literature_skips_when_no_atom_count_available():
    src = LiteratureSource(LITERATURE_DB)
    bridged = [_bridged_fm2()]
    verdict = src.check(bridged, PhysicalStateClaim())
    assert verdict.decision == SourceDecision.SKIP


def test_build_default_verifier_can_opt_in_to_energy_comparison():
    """The integrator-level flag plumbs through to the literature source."""
    v = build_default_verifier(
        literature_db_path=LITERATURE_DB,
        literature_compare_energy=True,
    )
    src = v._sources["literature"]
    assert src is not None
    assert src.compare_energy is True


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------


def test_integrator_runs_all_sources_under_v4():
    verifier = build_default_verifier(
        literature_db_path=LITERATURE_DB,
    )
    bridged = [_bridged_fm1(n_pred=7,
                            positions=[[0.0, 0.0]] * 7,
                            confidences=[3.0] * 7),
               _bridged_fm2(energy=-1.79),
               _bridged_fm3()]
    claim = PhysicalStateClaim(n_atoms=7, motif="triangular_disk", temperature=0.5)
    verdict = verifier.verify(bridged, claim, sources_config=SourcesConfig.for_ablation("V4"))
    assert isinstance(verdict, VerifierVerdict)
    sources_run = {
        v.source_name for v in verdict.source_verdicts
        if v.decision is not SourceDecision.SKIP
    }
    # rule_library, cross_fm, conformal, literature should all run.
    # simulator skips without positions.
    assert "rule_library" in sources_run
    assert "literature" in sources_run
    assert "cross_fm" in sources_run
    assert "conformal" in sources_run


def test_integrator_disabled_sources_skip_under_v0():
    verifier = build_default_verifier(literature_db_path=LITERATURE_DB)
    bridged = [_bridged_fm2()]
    verdict = verifier.verify(
        bridged, PhysicalStateClaim(),
        sources_config=SourcesConfig.for_ablation("V0"),
    )
    assert verdict.aggregate_decision is SourceDecision.SKIP
    for v in verdict.source_verdicts:
        assert v.decision is SourceDecision.SKIP


def test_integrator_v1_only_runs_rule_library():
    verifier = build_default_verifier(literature_db_path=LITERATURE_DB)
    bridged = [_bridged_fm2(energy=-1.42)]
    verdict = verifier.verify(
        bridged, PhysicalStateClaim(),
        sources_config=SourcesConfig.for_ablation("V1"),
    )
    by_name = {v.source_name: v for v in verdict.source_verdicts}
    assert by_name["rule_library"].decision is not SourceDecision.SKIP
    for n in ("literature", "cross_fm", "simulator", "conformal"):
        assert by_name[n].decision is SourceDecision.SKIP


def test_integrator_aggregates_fail_over_caveat():
    """A FAIL from any source overrides any number of CAVEATs."""
    verifier = build_default_verifier(literature_db_path=LITERATURE_DB)
    bridged = [_bridged_fm2(energy=-50.0)]  # hard fail on non_negativity
    verdict = verifier.verify(bridged, PhysicalStateClaim(n_atoms=7))
    assert verdict.aggregate_decision is SourceDecision.FAIL
    assert "rule_library" in verdict.hint.flagged_sources


def test_integrator_hint_lists_flagged_sources():
    verifier = build_default_verifier(literature_db_path=LITERATURE_DB)
    bridged = [_bridged_fm2(energy=-1.42)]
    claim = PhysicalStateClaim(per_atom_potential_energy=-2.0, n_atoms=7,
                               motif="triangular_disk")
    verdict = verifier.verify(bridged, claim)
    # conformal flags claim outside band -> caveat.
    assert verdict.aggregate_decision in (SourceDecision.CAVEAT, SourceDecision.FAIL)
    assert verdict.hint.flagged_sources


def test_verifier_verdict_json_round_trip():
    verifier = build_default_verifier(literature_db_path=LITERATURE_DB)
    bridged = [_bridged_fm2()]
    verdict = verifier.verify(bridged, PhysicalStateClaim(n_atoms=7))
    payload = verdict.model_dump_json()
    rehydrated = VerifierVerdict.model_validate_json(payload)
    assert rehydrated.model_dump() == verdict.model_dump()
