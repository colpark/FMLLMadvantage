"""Tests for the structure-preserving and language-anchored bridges.

Coverage:
    - FMContext loads from disk via compose.load_fm_context, falling
      back gracefully when probe_report.yaml or calibration.json is
      missing.
    - The three structure-preserving bridges produce well-formed
      BridgedFMOutput objects with the right typed value payloads,
      uncertainty fields, applicable constraints, and dependencies.
    - JSON round-trip preserves every BridgedFMOutput field.
    - The three language-anchored bridges produce captions whose
      numerical content matches the structure-preserving output
      after parse_caption.
    - Factory dispatch picks the correct subclass per FM name and
      raises on unknown names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fmllm.bridges import (
    FMContext,
    LanguageAnchoredBridge,
    StructurePreservingBridge,
    load_fm_context,
    make_language_bridge,
    make_structure_bridge,
    parse_caption,
)
from fmllm.bridges.compose import metadata_yaml_path
from fmllm.fms._schemas import (
    BridgedFMOutput,
    ProbeReport,
    ProbeResult,
    load_fm_metadata,
)
from fmllm.fms._schemas.probe_schema import now_utc_iso


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_context(fm_name: str, *, with_calibration: bool = True) -> FMContext:
    """Build an FMContext from the shipped metadata + a synthetic probe
    report and calibration."""
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
        fm_name=metadata.name,
        fm_version=metadata.version,
        timestamp_utc=now_utc_iso(),
        results=probe_results,
    )
    calibration: dict = {}
    if with_calibration:
        if fm_name == "fm1_image":
            cal = {"0.1000": 0.76, "0.2000": 0.49}
            score = "position_l2_lj"
        elif fm_name == "fm2_rdf":
            cal = {"0.1000": 0.07, "0.2000": 0.045}
            score = "energy_abs_residual"
        else:
            cal = {"0.1000": 1.19, "0.2000": 0.88}
            score = "ke_distribution_nll"
        calibration = {
            "fm_name": fm_name,
            "score_name": score,
            "thresholds": cal,
            "extra": {},
        }
    return FMContext(
        fm_name=fm_name,
        metadata=metadata,
        probe_report=probe_report,
        calibration=calibration,
    )


def _fm1_raw():
    torch.manual_seed(0)
    return {
        "count_logits": torch.cat([
            torch.full((30,), -3.0),
            torch.tensor([5.0]),  # argmax = 30 ... but max_n_atoms is 30 -> count_logits has length 31
        ]),
        "positions": torch.tensor([
            [0.5, 0.3],
            [-1.2, 0.7],
            [0.0, -1.5],
            [10.0, 10.0],  # low-confidence ghost
        ]),
        "confidence_logits": torch.tensor([3.0, 2.5, 1.5, -3.0]),
    }


def _fm2_raw():
    return {"energy": torch.tensor(-1.42)}


def _fm3_raw():
    return {"alpha": torch.tensor(2.0), "beta": torch.tensor(0.55)}


# ---------------------------------------------------------------------------
# FMContext + compose
# ---------------------------------------------------------------------------


def test_load_fm_context_handles_missing_artifacts(tmp_path: Path):
    """compose.load_fm_context falls back to empty probe report and
    empty calibration when the files are absent."""
    ctx = load_fm_context(fm_name="fm1_image", checkpoint_dir=tmp_path)
    assert ctx.fm_name == "fm1_image"
    assert ctx.probe_report.results == []
    assert ctx.calibration == {}


def test_load_fm_context_reads_real_artifacts(tmp_path: Path):
    """compose.load_fm_context reads probe_report.yaml and calibration.json
    when present in the checkpoint dir."""
    from fmllm.fms._schemas import save_probe_report, ProbeReport, ProbeResult
    from fmllm.fms.common import write_conformal_calibration

    pr = ProbeReport(
        fm_name="fm2_rdf", fm_version="0.1.0",
        timestamp_utc=now_utc_iso(),
        results=[ProbeResult(
            constraint_name="non_negativity",
            satisfaction_score=1.0, num_test_cases=10,
            metric="dummy", passes_threshold=True, threshold=0.99,
            details={},
        )],
    )
    save_probe_report(pr, tmp_path / "probe_report.yaml")
    write_conformal_calibration(
        tmp_path / "calibration.json",
        fm_name="fm2_rdf", score_name="energy_abs_residual",
        alpha_to_threshold={0.10: 0.07, 0.20: 0.045},
    )
    ctx = load_fm_context(fm_name="fm2_rdf", checkpoint_dir=tmp_path)
    assert len(ctx.probe_report.results) == 1
    assert ctx.calibration_threshold(0.10) == pytest.approx(0.07)


def test_calibration_threshold_lookup_returns_none_for_unknown_alpha():
    ctx = _build_context("fm1_image")
    assert ctx.calibration_threshold(0.05) is None
    assert ctx.calibration_threshold(0.10) == pytest.approx(0.76)


# ---------------------------------------------------------------------------
# StructurePreservingBridge
# ---------------------------------------------------------------------------


def test_fm1_structure_bridge_emits_bridged_output():
    ctx = _build_context("fm1_image")
    bridge = make_structure_bridge(ctx)
    out = bridge.emit(_fm1_raw(), input_provenance={"specimen_id": 7})
    assert isinstance(out, BridgedFMOutput)
    assert out.source.fm_name == "fm1_image"
    assert out.source.raw_input_provenance["specimen_id"] == 7
    assert out.prediction.quantity == "atom_positions_lj"
    assert out.prediction.units == "lj_units"
    # confidence threshold > 0.5 keeps three atoms (logits 3.0, 2.5, 1.5).
    assert out.prediction.value["n_atoms_pred"] == 30
    assert len(out.prediction.value["positions"]) == 3
    assert out.prediction.uncertainty is not None
    assert out.prediction.uncertainty.lower == 0.0
    assert out.prediction.uncertainty.upper == pytest.approx(0.76)
    assert out.prediction.uncertainty.confidence_level == 0.90
    # Probe scores show up as applicable constraints.
    names = {c.constraint_name for c in out.applicable_constraints}
    assert names == {"translation_equivariance", "atom_count_consistency", "positions_in_box"}
    # Dependencies are materialized for atom_count.
    deps = {d.target_variable: d for d in out.dependencies}
    assert "atom_count" in deps
    assert deps["atom_count"].derived_value == 30


def test_fm2_structure_bridge_emits_symmetric_uncertainty():
    ctx = _build_context("fm2_rdf")
    bridge = make_structure_bridge(ctx)
    out = bridge.emit(_fm2_raw())
    assert out.prediction.value["value_lj"] == pytest.approx(-1.42)
    assert out.prediction.units == "lj_per_atom"
    assert out.prediction.uncertainty is not None
    assert out.prediction.uncertainty.lower == pytest.approx(-1.42 - 0.07)
    assert out.prediction.uncertainty.upper == pytest.approx(-1.42 + 0.07)


def test_fm3_structure_bridge_packs_gamma_moments():
    ctx = _build_context("fm3_traj")
    bridge = make_structure_bridge(ctx)
    out = bridge.emit(_fm3_raw())
    val = out.prediction.value
    assert val["alpha"] == pytest.approx(2.0)
    assert val["beta"] == pytest.approx(0.55)
    assert val["mean"] == pytest.approx(2.0 * 0.55)
    assert val["variance"] == pytest.approx(2.0 * 0.55 * 0.55)
    assert val["implied_temperature_lj"] == pytest.approx(2.0 * 0.55)
    # FM3 uncertainty is None; the verifier checks NLL against calibration instead.
    assert out.prediction.uncertainty is None
    deps = {d.target_variable: d for d in out.dependencies}
    assert "temperature" in deps
    assert deps["temperature"].derived_value == pytest.approx(2.0 * 0.55)


def test_bridged_output_round_trips_through_json():
    """JSON dump + load preserves every field across all three FMs."""
    cases = [
        ("fm1_image", _fm1_raw()),
        ("fm2_rdf", _fm2_raw()),
        ("fm3_traj", _fm3_raw()),
    ]
    for fm_name, raw in cases:
        ctx = _build_context(fm_name)
        bridge = make_structure_bridge(ctx)
        out = bridge.emit(raw, input_provenance={"specimen_id": 1})
        payload = out.model_dump_json()
        rehydrated = BridgedFMOutput.model_validate(json.loads(payload))
        assert rehydrated.model_dump() == out.model_dump()


def test_structure_bridge_without_calibration_omits_uncertainty():
    ctx = _build_context("fm1_image", with_calibration=False)
    bridge = make_structure_bridge(ctx)
    out = bridge.emit(_fm1_raw())
    assert out.prediction.uncertainty is None


def test_structure_bridge_emits_in_distribution_flag():
    ctx = _build_context("fm2_rdf")
    bridge = make_structure_bridge(ctx)
    out = bridge.emit(_fm2_raw(), in_distribution=False)
    assert out.source.in_distribution is False


# ---------------------------------------------------------------------------
# LanguageAnchoredBridge
# ---------------------------------------------------------------------------


def test_fm1_language_caption_round_trip():
    ctx = _build_context("fm1_image")
    bridge = make_language_bridge(ctx)
    caption = bridge.emit(_fm1_raw(), input_provenance={"specimen_id": 7})
    assert "image shows" in caption
    parsed = parse_caption(caption, "fm1_image")
    assert parsed["n_atoms_pred"] == 30
    assert len(parsed["positions"]) == 3
    # First position from the raw output: (0.5, 0.3).
    assert parsed["positions"][0] == pytest.approx((0.5, 0.3), abs=1e-3)
    assert len(parsed["confidences"]) == 3


def test_fm2_language_caption_round_trip():
    ctx = _build_context("fm2_rdf")
    bridge = make_language_bridge(ctx)
    caption = bridge.emit(_fm2_raw())
    parsed = parse_caption(caption, "fm2_rdf")
    assert parsed["energy_lj_per_atom"] == pytest.approx(-1.42, abs=1e-4)
    assert parsed["band_sigma"] == pytest.approx(0.07, abs=1e-4)


def test_fm3_language_caption_round_trip():
    ctx = _build_context("fm3_traj")
    bridge = make_language_bridge(ctx)
    caption = bridge.emit(_fm3_raw())
    parsed = parse_caption(caption, "fm3_traj")
    assert parsed["alpha"] == pytest.approx(2.0, abs=1e-4)
    assert parsed["beta"] == pytest.approx(0.55, abs=1e-4)
    assert parsed["temperature_lj"] == pytest.approx(2.0 * 0.55, abs=1e-4)


def test_language_bridge_includes_constraint_summary():
    ctx = _build_context("fm2_rdf")
    bridge = make_language_bridge(ctx)
    caption = bridge.emit(_fm2_raw())
    assert "Constraints:" in caption
    for cname in ("permutation_invariance", "extensive_scaling", "non_negativity"):
        assert cname in caption


def test_language_bridge_marks_out_of_distribution():
    ctx = _build_context("fm2_rdf")
    bridge = make_language_bridge(ctx)
    caption = bridge.emit(_fm2_raw(), in_distribution=False)
    assert "out-of-distribution" in caption


def test_language_bridge_handles_no_calibration():
    ctx = _build_context("fm2_rdf", with_calibration=False)
    bridge = make_language_bridge(ctx)
    caption = bridge.emit(_fm2_raw())
    assert "plus or minus" not in caption


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def test_factories_dispatch_by_fm_name():
    for fm in ("fm1_image", "fm2_rdf", "fm3_traj"):
        ctx = _build_context(fm)
        s = make_structure_bridge(ctx)
        l_ = make_language_bridge(ctx)
        assert isinstance(s, StructurePreservingBridge)
        assert isinstance(l_, LanguageAnchoredBridge)
        assert s.fm_name == fm
        assert l_.fm_name == fm


def test_factories_reject_unknown_fm_name():
    metadata = load_fm_metadata(metadata_yaml_path("fm1_image"))
    ctx = FMContext(
        fm_name="fm99_made_up",
        metadata=metadata,
        probe_report=ProbeReport(
            fm_name="fm99_made_up", fm_version="0.0.1",
            timestamp_utc=now_utc_iso(), results=[],
        ),
        calibration={},
    )
    with pytest.raises(ValueError, match="no structure-preserving bridge"):
        make_structure_bridge(ctx)
    with pytest.raises(ValueError, match="no language-anchored bridge"):
        make_language_bridge(ctx)


def test_parse_caption_rejects_unknown_fm_name():
    with pytest.raises(ValueError, match="unknown fm_name"):
        parse_caption("anything", "fm99_made_up")
