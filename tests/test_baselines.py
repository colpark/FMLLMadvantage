"""Tests for the Phase 8a baselines and the goal-accuracy metric.

These tests build trajectories by hand (the OHVD loop and FM
runtime are not exercised here) and verify:

* `NoOpVerifier` returns aggregate PASS with one stub source verdict.
* `run_naked_baseline` produces a one-step Trajectory with the
  expected schema, including parse-failure handling for non-commit
  responses.
* `evaluation.accuracy.measure(...)` correctly computes per-field
  accuracy, compound accuracy, hallucination rate, and calibrated
  abstention rate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from fmllm.baselines import NoOpVerifier, run_naked_baseline
from fmllm.evaluation import accuracy
from fmllm.orchestrator import (
    ActionType,
    LLMAction,
    MockLLM,
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


def _verdict(decision: SourceDecision) -> VerifierVerdict:
    return VerifierVerdict(
        aggregate_decision=decision,
        source_verdicts=[],
        timestamp=_now(),
    )


def _commit_trajectory(
    *,
    specimen_id: int,
    claim: PhysicalStateClaim,
    decision: SourceDecision = SourceDecision.PASS,
) -> Trajectory:
    return Trajectory(
        run_id=f"r{specimen_id}",
        query="",
        specimen_id=specimen_id,
        started_utc=_now(),
        finished_utc=_now(),
        termination=TerminationReason.COMMITTED,
        final_claim=claim,
        final_verdict=_verdict(decision),
        steps=[
            Step(
                step_index=0,
                step_type=StepType.FINAL,
                timestamp_utc=_now(),
                llm_action=LLMAction(
                    action_type=ActionType.COMMIT, raw_text="",
                ),
                claim=claim,
                verdict=_verdict(decision),
            )
        ],
    )


# ---------------------------------------------------------------------------
# NoOpVerifier
# ---------------------------------------------------------------------------


def test_noop_verifier_returns_pass_with_stub_source():
    v = NoOpVerifier()
    claim = PhysicalStateClaim(n_atoms=7)
    verdict = v.verify([], claim)
    assert verdict.aggregate_decision is SourceDecision.PASS
    assert len(verdict.source_verdicts) == 1
    assert verdict.source_verdicts[0].source_name == "noop"
    assert verdict.source_verdicts[0].decision is SourceDecision.PASS


def test_noop_verifier_ignores_sources_config():
    v = NoOpVerifier()
    from fmllm.verifier.schema import SourcesConfig
    cfg = SourcesConfig(rule_library=False, literature=False, cross_fm=False, simulator=False, conformal=False)
    verdict = v.verify([], PhysicalStateClaim(), sources_config=cfg)
    assert verdict.aggregate_decision is SourceDecision.PASS


# ---------------------------------------------------------------------------
# Naked baseline
# ---------------------------------------------------------------------------


def test_run_naked_baseline_commit_path():
    response = (
        '{"action": "commit", "claim": {"n_atoms": 7, "temperature": 0.5, '
        '"motif": "triangular_disk", "per_atom_potential_energy": -1.5}}'
    )
    llm = MockLLM([response], cycle=True)
    traj = run_naked_baseline(
        llm=llm, query="identify the specimen", specimen_id=42,
    )
    assert traj.specimen_id == 42
    assert traj.termination is TerminationReason.COMMITTED
    assert traj.final_claim is not None
    assert traj.final_claim.n_atoms == 7
    assert traj.final_verdict is None  # B0 has no verifier
    assert len(traj.steps) == 1
    assert traj.steps[0].step_type is StepType.FINAL
    assert traj.metadata["baseline"] == "naked"


def test_run_naked_baseline_parse_failure_path():
    llm = MockLLM(["this is not JSON"], cycle=True)
    traj = run_naked_baseline(
        llm=llm, query="identify the specimen", specimen_id=42,
    )
    assert traj.termination is TerminationReason.PARSE_FAILURE
    assert traj.final_claim is None
    assert traj.steps[0].step_type is StepType.ERROR


def test_run_naked_baseline_treats_hypothesize_as_parse_failure():
    """The naked baseline expects a single commit; hypothesize is rejected."""
    llm = MockLLM([
        '{"action": "hypothesize", "claim": {"n_atoms": 7}}'
    ], cycle=True)
    traj = run_naked_baseline(
        llm=llm, query="identify the specimen", specimen_id=42,
    )
    assert traj.termination is TerminationReason.PARSE_FAILURE


# ---------------------------------------------------------------------------
# Goal-accuracy metric
# ---------------------------------------------------------------------------


def test_accuracy_compound_with_perfect_claims():
    truth = {
        100: {"n": 7, "t": 0.5, "motif": "triangular_disk"},
        101: {"n": 12, "t": 1.0, "motif": "ring"},
    }
    trajs = [
        _commit_trajectory(
            specimen_id=100,
            claim=PhysicalStateClaim(
                n_atoms=7, temperature=0.5, motif="triangular_disk",
            ),
        ),
        _commit_trajectory(
            specimen_id=101,
            claim=PhysicalStateClaim(
                n_atoms=12, temperature=1.0, motif="ring",
            ),
        ),
    ]
    r = accuracy.measure(trajectories=trajs, truth=truth)
    assert r.skipped is False
    assert r.metric_value == pytest.approx(1.0, abs=1.0e-6)
    assert r.passes is True
    assert r.details["n_committed"] == 2
    assert r.details["per_field_accuracy"] == {
        "n_atoms": 1.0, "temperature": 1.0, "motif": 1.0,
    }


def test_accuracy_handles_partial_correctness():
    truth = {
        100: {"n": 7, "t": 0.5, "motif": "triangular_disk"},
        101: {"n": 12, "t": 1.0, "motif": "ring"},
    }
    # Specimen 100: motif wrong. Specimen 101: T off by 50%.
    trajs = [
        _commit_trajectory(
            specimen_id=100,
            claim=PhysicalStateClaim(
                n_atoms=7, temperature=0.5, motif="linear",
            ),
        ),
        _commit_trajectory(
            specimen_id=101,
            claim=PhysicalStateClaim(
                n_atoms=12, temperature=1.5, motif="ring",
            ),
        ),
    ]
    r = accuracy.measure(trajectories=trajs, truth=truth)
    assert r.skipped is False
    assert r.metric_value == pytest.approx(0.0, abs=1.0e-6)  # both compound-fail
    assert r.details["per_field_accuracy"]["n_atoms"] == 1.0
    assert r.details["per_field_accuracy"]["temperature"] == 0.5
    assert r.details["per_field_accuracy"]["motif"] == 0.5


def test_accuracy_hallucination_rate():
    """Two trajectories: one PASS+correct, one PASS+wrong → halluc=0.5."""
    truth = {
        100: {"n": 7, "t": 0.5, "motif": "triangular_disk"},
        101: {"n": 12, "t": 1.0, "motif": "ring"},
    }
    trajs = [
        _commit_trajectory(
            specimen_id=100,
            claim=PhysicalStateClaim(
                n_atoms=7, temperature=0.5, motif="triangular_disk",
            ),
            decision=SourceDecision.PASS,
        ),
        _commit_trajectory(
            specimen_id=101,
            claim=PhysicalStateClaim(
                n_atoms=4, temperature=0.1, motif="linear",
            ),
            decision=SourceDecision.PASS,
        ),
    ]
    r = accuracy.measure(trajectories=trajs, truth=truth)
    assert r.details["hallucination_rate"] == pytest.approx(0.5, abs=1.0e-6)


def test_accuracy_calibrated_abstention_rate():
    """Two wrong commits, one CAVEAT, one PASS → abstention=0.5."""
    truth = {
        100: {"n": 7, "t": 0.5, "motif": "triangular_disk"},
        101: {"n": 12, "t": 1.0, "motif": "ring"},
    }
    trajs = [
        _commit_trajectory(
            specimen_id=100,
            claim=PhysicalStateClaim(n_atoms=4, temperature=0.5, motif="linear"),
            decision=SourceDecision.CAVEAT,
        ),
        _commit_trajectory(
            specimen_id=101,
            claim=PhysicalStateClaim(n_atoms=4, temperature=0.5, motif="linear"),
            decision=SourceDecision.PASS,
        ),
    ]
    r = accuracy.measure(trajectories=trajs, truth=truth)
    assert r.details["calibrated_abstention_rate"] == pytest.approx(0.5, abs=1.0e-6)


def test_accuracy_skipped_when_no_commits():
    truth = {100: {"n": 7, "t": 0.5, "motif": "triangular_disk"}}
    trajs = [
        Trajectory(
            run_id="r0",
            query="",
            specimen_id=100,
            started_utc=_now(),
            finished_utc=_now(),
            termination=TerminationReason.PARSE_FAILURE,
            steps=[],
        )
    ]
    r = accuracy.measure(trajectories=trajs, truth=truth)
    assert r.skipped is True


def test_accuracy_n_total_includes_unparseable_trajectories():
    """commit_rate counts committed / total, where total includes unparsed."""
    truth = {100: {"n": 7, "t": 0.5, "motif": "triangular_disk"}}
    good = _commit_trajectory(
        specimen_id=100,
        claim=PhysicalStateClaim(n_atoms=7, temperature=0.5, motif="triangular_disk"),
    )
    bad = Trajectory(
        run_id="r1",
        query="",
        specimen_id=101,
        started_utc=_now(),
        finished_utc=_now(),
        termination=TerminationReason.PARSE_FAILURE,
        steps=[],
    )
    truth[101] = {"n": 12, "t": 1.0, "motif": "ring"}
    r = accuracy.measure(trajectories=[good, bad], truth=truth)
    assert r.details["n_total"] == 2
    assert r.details["n_committed"] == 1
    assert r.details["commit_rate"] == pytest.approx(0.5, abs=1.0e-6)
