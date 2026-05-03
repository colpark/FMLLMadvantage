"""Tests for the eight world-model evaluation tests.

These tests build small synthetic trajectory lists and check that
each :func:`measure` returns sensible :class:`TestResult` objects:
correct pass/fail flags, correct skipped behavior on degenerate input,
correct extraction of evidence from per-step verdicts.

The tests do not load any FM weights or HDF5 data; ground-truth
dictionaries are constructed by hand. The :func:`build_trajectory`
helper assembles trajectories from a compact spec.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from fmllm.evaluation import (
    calibrated_uncertainty,
    federated_factorability,
    goal_competence,
    prediction_compression,
    prediction_distinction,
    step_recoverability,
    trajectory_compression,
    trajectory_distinction,
)
from fmllm.evaluation.schema import (
    TestResult,
    make_skipped,
    threshold_check,
)
from fmllm.evaluation.utils import (
    claim_distance,
    edit_distance,
    physical_equivalence_class,
)
from fmllm.fms._schemas import BridgedFMOutput, Prediction, Source
from fmllm.orchestrator import (
    ActionType,
    LLMAction,
    Step,
    StepType,
    TerminationReason,
    Trajectory,
)
from fmllm.verifier.schema import (
    PhysicalStateClaim,
    SourceDecision,
    SourceVerdict,
    VerifierVerdict,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _bridged_output(
    fm_name: str, value: dict[str, Any], in_distribution: bool = True,
) -> BridgedFMOutput:
    return BridgedFMOutput(
        source=Source(
            fm_name=fm_name,
            fm_version="test",
            in_distribution=in_distribution,
            raw_input_provenance={},
        ),
        prediction=Prediction(
            quantity=fm_name,
            value=value,
            units="lj_units",
            uncertainty=None,
        ),
        applicable_constraints=[],
        dependencies=[],
        timestamp=_now(),
    )


def _verdict(
    decision: SourceDecision,
    *,
    sources: list[SourceVerdict] | None = None,
) -> VerifierVerdict:
    return VerifierVerdict(
        aggregate_decision=decision,
        source_verdicts=sources or [],
        timestamp=_now(),
    )


def build_trajectory(
    *,
    run_id: str,
    specimen_id: int,
    actions: list[str],
    final_claim: PhysicalStateClaim | None = None,
    fm_outputs: dict[str, dict[str, Any]] | None = None,
    aggregate_decision: SourceDecision = SourceDecision.PASS,
    conformal_per_fm: list[dict[str, Any]] | None = None,
) -> Trajectory:
    """Build a compact trajectory from an action sequence.

    ``actions`` is a list of action keys: "call_fm:<name>", "hypothesize",
    "commit". For each ``call_fm:<name>`` step a bridged output is
    inserted using ``fm_outputs[name]`` (default empty dict).
    """
    fm_outputs = fm_outputs or {}
    conformal_per_fm = conformal_per_fm or []
    steps: list[Step] = []
    idx = 0

    for a in actions:
        if a.startswith("call_fm:"):
            fm_name = a.split(":", 1)[1]
            value = fm_outputs.get(fm_name, {})
            steps.append(
                Step(
                    step_index=idx,
                    step_type=StepType.OBSERVATION,
                    timestamp_utc=_now(),
                    llm_action=LLMAction(action_type=ActionType.CALL_FM, raw_text=""),
                    bridged_output=_bridged_output(fm_name, value),
                )
            )
            idx += 1
        elif a == "hypothesize":
            steps.append(
                Step(
                    step_index=idx,
                    step_type=StepType.HYPOTHESIS,
                    timestamp_utc=_now(),
                    llm_action=LLMAction(
                        action_type=ActionType.HYPOTHESIZE, raw_text="",
                    ),
                    claim=final_claim,
                )
            )
            idx += 1
        elif a == "commit":
            sources: list[SourceVerdict] = []
            if conformal_per_fm:
                sources.append(
                    SourceVerdict(
                        source_name="conformal",
                        decision=SourceDecision.PASS,
                        confidence=1.0,
                        message="",
                        evidence={"per_fm": conformal_per_fm},
                    )
                )
            steps.append(
                Step(
                    step_index=idx,
                    step_type=StepType.FINAL,
                    timestamp_utc=_now(),
                    llm_action=LLMAction(action_type=ActionType.COMMIT, raw_text=""),
                    claim=final_claim,
                    verdict=_verdict(aggregate_decision, sources=sources),
                )
            )
            idx += 1
        else:
            raise ValueError(f"unknown action {a!r}")

    final_verdict = (
        _verdict(aggregate_decision) if any(a == "commit" for a in actions) else None
    )
    return Trajectory(
        run_id=run_id,
        query="",
        specimen_id=specimen_id,
        started_utc=_now(),
        finished_utc=_now(),
        termination=TerminationReason.COMMITTED if final_verdict else TerminationReason.BUDGET_EXHAUSTED,
        final_claim=final_claim,
        final_verdict=final_verdict,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def test_threshold_check_directions():
    assert threshold_check(0.9, 0.8, "ge") is True
    assert threshold_check(0.7, 0.8, "ge") is False
    assert threshold_check(0.1, 0.2, "le") is True
    assert threshold_check(0.5, 0.45, "eq") is True
    assert threshold_check(0.5, 0.30, "eq") is False
    with pytest.raises(ValueError):
        threshold_check(0.5, 0.5, "neq")


def test_make_skipped_marks_correctly():
    r = make_skipped(
        test_name="x",
        layer="trajectory",
        metric_name="m",
        threshold=0.5,
        reason="no data",
    )
    assert r.skipped is True
    assert r.passes is False
    assert r.skip_reason == "no data"


def test_claim_distance_matches_intuition():
    a = PhysicalStateClaim(n_atoms=7, temperature=0.5, motif="triangle")
    b = PhysicalStateClaim(n_atoms=7, temperature=0.5, motif="triangle")
    assert claim_distance(a, b) == 0
    c = PhysicalStateClaim(n_atoms=8, temperature=0.5, motif="triangle")
    assert claim_distance(a, c) == 1
    d = PhysicalStateClaim(n_atoms=7, temperature=0.5, motif="square")
    assert claim_distance(a, d) == 1


def test_edit_distance_basic():
    assert edit_distance(("a", "b"), ("a", "b")) == 0
    assert edit_distance(("a", "b"), ("a",)) == 1
    assert edit_distance((), ("a",)) == 1
    assert edit_distance(("a",), ("b",)) == 1


def test_physical_equivalence_class_ignores_temperature():
    t1 = {"n": 7, "t": 0.5, "motif": "triangle"}
    t2 = {"n": 7, "t": 1.5, "motif": "triangle"}
    assert physical_equivalence_class(t1) == physical_equivalence_class(t2)


# ---------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------


def test_trajectory_compression_passes_on_clones():
    claim = PhysicalStateClaim(n_atoms=7, temperature=0.5, motif="triangle")
    trajs = [
        build_trajectory(
            run_id=f"r{i}", specimen_id=100 + i,
            actions=["call_fm:fm1_image", "commit"],
            final_claim=claim,
        )
        for i in range(3)
    ]
    truth = {
        100: {"n": 7, "t": 0.5, "motif": "triangle"},
        101: {"n": 7, "t": 0.5, "motif": "triangle"},
        102: {"n": 7, "t": 0.5, "motif": "triangle"},
    }
    r = trajectory_compression.measure(trajectories=trajs, truth=truth)
    assert isinstance(r, TestResult)
    assert r.skipped is False
    assert r.passes is True
    assert r.metric_value is not None and r.metric_value <= 2.0


def test_trajectory_compression_skips_when_no_class_has_pairs():
    claim = PhysicalStateClaim(n_atoms=7)
    trajs = [
        build_trajectory(
            run_id="r0", specimen_id=100,
            actions=["commit"], final_claim=claim,
        ),
        build_trajectory(
            run_id="r1", specimen_id=101,
            actions=["commit"], final_claim=claim,
        ),
    ]
    truth = {
        100: {"n": 7, "t": 0.5, "motif": "triangle"},
        101: {"n": 8, "t": 0.5, "motif": "square"},
    }
    r = trajectory_compression.measure(trajectories=trajs, truth=truth)
    assert r.skipped is True


def test_trajectory_distinction_separates_classes():
    a_claim = PhysicalStateClaim(n_atoms=7, motif="triangle")
    b_claim = PhysicalStateClaim(n_atoms=20, motif="square")
    trajs = []
    for i in range(3):
        trajs.append(build_trajectory(
            run_id=f"a{i}", specimen_id=100 + i,
            actions=["call_fm:fm1_image", "commit"],
            final_claim=a_claim,
        ))
        trajs.append(build_trajectory(
            run_id=f"b{i}", specimen_id=200 + i,
            actions=["call_fm:fm1_image", "call_fm:fm2_rdf", "commit"],
            final_claim=b_claim,
        ))
    truth: dict[int, dict[str, Any]] = {}
    for i in range(3):
        truth[100 + i] = {"n": 7, "t": 0.5, "motif": "triangle"}
        truth[200 + i] = {"n": 20, "t": 0.5, "motif": "square"}
    r = trajectory_distinction.measure(
        trajectories=trajs, truth=truth, n_pairs=20,
    )
    assert r.skipped is False
    assert r.metric_value is not None
    assert r.metric_value > 0


def test_step_recoverability_passes_when_claim_matches_fm_signals():
    claim = PhysicalStateClaim(
        n_atoms=8, temperature=0.5, per_atom_potential_energy=-1.0,
    )
    trajs = [
        build_trajectory(
            run_id="r0", specimen_id=100,
            actions=[
                "call_fm:fm1_image", "call_fm:fm2_rdf", "call_fm:fm3_traj",
                "commit",
            ],
            final_claim=claim,
            fm_outputs={
                "fm1_image": {"n_atoms_pred": 7},          # off-by-one OK
                "fm2_rdf": {"value_lj": -1.2},             # within 0.5
                "fm3_traj": {"alpha": 1.0, "beta": 0.45},  # 0.45 vs 0.5: 10% rel
            },
        )
    ]
    r = step_recoverability.measure(trajectories=trajs, threshold=0.50)
    assert r.skipped is False
    assert r.passes is True
    assert r.n_samples >= 3


def test_step_recoverability_skips_when_no_observations():
    claim = PhysicalStateClaim(n_atoms=8)
    trajs = [
        build_trajectory(
            run_id="r0", specimen_id=100,
            actions=["commit"], final_claim=claim,
        )
    ]
    r = step_recoverability.measure(trajectories=trajs)
    assert r.skipped is True


# ---------------------------------------------------------------------------
# Layer 2
# ---------------------------------------------------------------------------


def test_prediction_compression_clusters_within_class():
    claim_a = PhysicalStateClaim(n_atoms=7, temperature=0.5, motif="triangle")
    trajs = [
        build_trajectory(
            run_id=f"r{i}", specimen_id=100 + i,
            actions=["commit"], final_claim=claim_a,
        )
        for i in range(3)
    ]
    truth = {
        100: {"n": 7, "t": 0.5, "motif": "triangle"},
        101: {"n": 7, "t": 0.6, "motif": "triangle"},
        102: {"n": 7, "t": 0.4, "motif": "triangle"},
    }
    r = prediction_compression.measure(trajectories=trajs, truth=truth)
    assert r.skipped is False
    assert r.passes is True


def test_prediction_distinction_separates_distant_classes():
    claim_a = PhysicalStateClaim(n_atoms=7, motif="triangle")
    claim_b = PhysicalStateClaim(n_atoms=25, motif="square")
    trajs = []
    for i in range(3):
        trajs.append(build_trajectory(
            run_id=f"a{i}", specimen_id=100 + i,
            actions=["commit"], final_claim=claim_a,
        ))
        trajs.append(build_trajectory(
            run_id=f"b{i}", specimen_id=200 + i,
            actions=["commit"], final_claim=claim_b,
        ))
    truth: dict[int, dict[str, Any]] = {}
    for i in range(3):
        truth[100 + i] = {"n": 7, "t": 0.5, "motif": "triangle"}
        truth[200 + i] = {"n": 25, "t": 0.5, "motif": "square"}
    r = prediction_distinction.measure(
        trajectories=trajs, truth=truth, n_pairs=20, threshold=2.0,
    )
    assert r.skipped is False
    assert r.metric_value is not None
    assert r.passes is True
    assert r.metric_value >= 2.0


def test_goal_competence_counts_per_goal_correctly():
    claim_perfect = PhysicalStateClaim(n_atoms=7, temperature=0.5, motif="triangle")
    claim_wrong_motif = PhysicalStateClaim(n_atoms=7, temperature=0.5, motif="square")
    trajs = [
        build_trajectory(
            run_id="r0", specimen_id=100,
            actions=["commit"], final_claim=claim_perfect,
        ),
        build_trajectory(
            run_id="r1", specimen_id=101,
            actions=["commit"], final_claim=claim_wrong_motif,
        ),
    ]
    truth = {
        100: {"n": 7, "t": 0.5, "motif": "triangle"},
        101: {"n": 7, "t": 0.5, "motif": "triangle"},
    }
    r = goal_competence.measure(trajectories=trajs, truth=truth)
    assert r.skipped is False
    # one of two trajectories misses motif. n_atoms (2/2), temperature (2/2),
    # motif (1/2), size_and_motif (1/2) -> 6/8 passed = 0.25 failure rate.
    assert r.metric_value == pytest.approx(0.25, abs=1.0e-6)
    assert r.passes is True


# ---------------------------------------------------------------------------
# Cross-layer
# ---------------------------------------------------------------------------


def test_federated_factorability_skips_with_one_ablation():
    trajs = [
        build_trajectory(
            run_id="r0", specimen_id=100,
            actions=["commit"],
            final_claim=PhysicalStateClaim(n_atoms=7),
            aggregate_decision=SourceDecision.PASS,
        )
    ]
    r = federated_factorability.measure(trajectories_by_ablation={"V0": trajs})
    assert r.skipped is True


def test_federated_factorability_passes_on_monotone_lattice():
    def trajs_for(rate: float) -> list[Trajectory]:
        out = []
        for i in range(20):
            decision = (
                SourceDecision.PASS if i / 20 < rate else SourceDecision.FAIL
            )
            out.append(build_trajectory(
                run_id=f"r{i}", specimen_id=i,
                actions=["commit"],
                final_claim=PhysicalStateClaim(n_atoms=7),
                aggregate_decision=decision,
            ))
        return out

    r = federated_factorability.measure(
        trajectories_by_ablation={
            "V0": trajs_for(0.10),
            "V1": trajs_for(0.30),
            "V2": trajs_for(0.50),
            "V3": trajs_for(0.70),
            "V4": trajs_for(0.90),
        }
    )
    assert r.skipped is False
    # 4 monotonic transitions, each gain 0.20 -> step_factor = 0.25.
    assert r.details["monotonicity"] == pytest.approx(1.0, abs=1.0e-6)
    assert r.details["step_factor"] == pytest.approx(0.25, abs=1.0e-6)
    assert r.passes is True


def test_federated_factorability_fails_on_brittle_lattice():
    def trajs_for(rate: float) -> list[Trajectory]:
        out = []
        for i in range(20):
            decision = (
                SourceDecision.PASS if i / 20 < rate else SourceDecision.FAIL
            )
            out.append(build_trajectory(
                run_id=f"r{i}", specimen_id=i,
                actions=["commit"],
                final_claim=PhysicalStateClaim(n_atoms=7),
                aggregate_decision=decision,
            ))
        return out

    # all gain concentrated in one transition.
    r = federated_factorability.measure(
        trajectories_by_ablation={
            "V0": trajs_for(0.10),
            "V1": trajs_for(0.10),
            "V2": trajs_for(0.10),
            "V3": trajs_for(0.10),
            "V4": trajs_for(0.90),
        }
    )
    assert r.passes is False
    assert r.details["step_factor"] == pytest.approx(1.0, abs=1.0e-6)


def test_calibrated_uncertainty_reads_per_fm_flags():
    claim = PhysicalStateClaim(n_atoms=7)
    # 18/20 ok per FM -> empirical = 0.9 -> gap = 0 -> pass.
    trajs = []
    per_fm_ok = [
        {"fm_name": "fm1_image", "in_distribution": True, "uncertainty_present": True, "flag": "ok"},
        {"fm_name": "fm2_rdf",   "in_distribution": True, "uncertainty_present": True, "flag": "ok"},
    ]
    per_fm_bad = [
        {"fm_name": "fm1_image", "in_distribution": False, "uncertainty_present": True, "flag": "out_of_distribution"},
        {"fm_name": "fm2_rdf",   "in_distribution": False, "uncertainty_present": True, "flag": "out_of_distribution"},
    ]
    for i in range(20):
        per_fm = per_fm_ok if i < 18 else per_fm_bad
        trajs.append(build_trajectory(
            run_id=f"r{i}", specimen_id=i,
            actions=["commit"], final_claim=claim,
            conformal_per_fm=per_fm,
        ))
    r = calibrated_uncertainty.measure(trajectories=trajs)
    assert r.skipped is False
    assert r.metric_value == pytest.approx(0.0, abs=1.0e-6)
    assert r.passes is True


def test_calibrated_uncertainty_skipped_without_conformal():
    claim = PhysicalStateClaim(n_atoms=7)
    trajs = [
        build_trajectory(
            run_id="r0", specimen_id=100,
            actions=["commit"], final_claim=claim,
        )
    ]
    r = calibrated_uncertainty.measure(trajectories=trajs)
    assert r.skipped is True
