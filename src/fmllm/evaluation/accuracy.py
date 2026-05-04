"""Ground-truth accuracy metrics for the baseline comparison.

The eight world-model tests measure *internal* coherence properties
(does the system behave like it has a world model?). They do not
compare claims to ground truth. This module fills that gap with
direct accuracy comparisons against the dataset HDF5:

* **N-accuracy**: ``|claimed - true| <= 2`` (off-by-2 tolerance,
  consistent with the rule library's atom_count_consistency).
* **T-accuracy**: ``|claimed - true| / true <= 25%`` rel.
* **Motif-accuracy**: exact string match.
* **Compound accuracy**: all three above hold simultaneously.
* **Commit rate**: fraction of trajectories that committed at all.
* **Hallucination rate**: fraction of committed trajectories that
  are wrong AND aggregated to ``PASS`` (the system was confident
  AND wrong).
* **Calibrated abstention**: when wrong, did the verifier flag it?
  Computed as ``CAVEAT-on-wrong / total-wrong``. Higher is better.

Each row in the returned :class:`fmllm.evaluation.schema.TestResult`
has ``test_name="goal_accuracy"``, ``layer="accuracy"``,
``metric_name="compound_accuracy"`` (the headline), and per-field
breakdowns under ``details``.

Depends on:
    fmllm.evaluation.schema, fmllm.orchestrator, fmllm.verifier.schema.
"""

from __future__ import annotations

from typing import Any

from fmllm.evaluation.schema import (
    TestResult,
    make_skipped,
    threshold_check,
)
from fmllm.orchestrator import Trajectory


def _check_n(claim, truth, tol: int) -> bool | None:
    if claim is None or claim.n_atoms is None:
        return None
    return abs(int(claim.n_atoms) - int(truth["n"])) <= tol


def _check_t(claim, truth, rel_tol: float) -> bool | None:
    if claim is None or claim.temperature is None:
        return None
    denom = max(abs(truth["t"]), 1.0e-3)
    return abs(claim.temperature - truth["t"]) / denom <= rel_tol


def _check_motif(claim, truth) -> bool | None:
    if claim is None or claim.motif is None:
        return None
    return claim.motif == truth["motif"]


def _aggregate_pass(t: Trajectory) -> str | None:
    """Return ``"pass"``, ``"caveat"``, ``"fail"``, or ``None`` (no verdict)."""
    if t.final_verdict is None:
        return None
    return t.final_verdict.aggregate_decision.value


def measure(
    *,
    trajectories: list[Trajectory],
    truth: dict[int, dict[str, Any]],
    threshold: float = 0.50,
    n_atoms_tolerance: int = 2,
    temperature_rel_tolerance: float = 0.25,
) -> TestResult:
    """Compute ground-truth accuracy across a baseline's trajectories.

    The headline metric is **compound_accuracy**: fraction of
    committed claims that get N, T, and motif all correct under the
    chosen tolerances. Default threshold ``>= 0.50`` (the baseline
    must beat coin-flip on the joint goal).
    """
    n_total = len(trajectories)
    n_committed = 0
    n_correct_compound = 0
    per_field = {
        "n_atoms": {"checked": 0, "passed": 0},
        "temperature": {"checked": 0, "passed": 0},
        "motif": {"checked": 0, "passed": 0},
    }
    n_pass_committed = 0
    n_caveat_committed = 0
    n_pass_and_wrong = 0       # hallucination
    n_caveat_and_wrong = 0     # calibrated abstention numerator
    n_wrong_committed = 0      # calibrated abstention denominator

    for t in trajectories:
        if t.specimen_id is None or t.specimen_id not in truth:
            continue
        gt = truth[t.specimen_id]
        claim = t.final_claim
        if claim is None:
            continue
        n_committed += 1

        n_ok = _check_n(claim, gt, n_atoms_tolerance)
        t_ok = _check_t(claim, gt, temperature_rel_tolerance)
        m_ok = _check_motif(claim, gt)

        if n_ok is not None:
            per_field["n_atoms"]["checked"] += 1
            if n_ok:
                per_field["n_atoms"]["passed"] += 1
        if t_ok is not None:
            per_field["temperature"]["checked"] += 1
            if t_ok:
                per_field["temperature"]["passed"] += 1
        if m_ok is not None:
            per_field["motif"]["checked"] += 1
            if m_ok:
                per_field["motif"]["passed"] += 1

        compound_ok = (
            n_ok is True and t_ok is True and m_ok is True
        )
        if compound_ok:
            n_correct_compound += 1
        else:
            n_wrong_committed += 1

        agg = _aggregate_pass(t)
        if agg == "pass":
            n_pass_committed += 1
            if not compound_ok:
                n_pass_and_wrong += 1
        elif agg == "caveat":
            n_caveat_committed += 1
            if not compound_ok:
                n_caveat_and_wrong += 1

    if n_committed == 0:
        return make_skipped(
            test_name="goal_accuracy",
            layer="accuracy",
            metric_name="compound_accuracy",
            threshold=threshold,
            threshold_direction="ge",
            reason="no committed trajectories with ground truth",
        )

    compound_acc = n_correct_compound / n_committed
    per_field_rate = {
        k: (v["passed"] / v["checked"]) if v["checked"] else None
        for k, v in per_field.items()
    }
    commit_rate = n_committed / n_total if n_total else 0.0
    hallucination_rate = (
        n_pass_and_wrong / n_pass_committed if n_pass_committed else None
    )
    calibrated_abstention = (
        n_caveat_and_wrong / n_wrong_committed if n_wrong_committed else None
    )

    passes = threshold_check(compound_acc, threshold, "ge")
    return TestResult(
        test_name="goal_accuracy",
        layer="accuracy",
        metric_name="compound_accuracy",
        metric_value=compound_acc,
        threshold=threshold,
        threshold_direction="ge",
        passes=passes,
        n_samples=n_committed,
        details={
            "n_total": n_total,
            "n_committed": n_committed,
            "commit_rate": commit_rate,
            "compound_correct": n_correct_compound,
            "per_field_accuracy": per_field_rate,
            "per_field_counts": per_field,
            "tolerances": {
                "n_atoms": n_atoms_tolerance,
                "temperature_rel": temperature_rel_tolerance,
                "motif": "exact",
            },
            "verdict_breakdown": {
                "pass": n_pass_committed,
                "caveat": n_caveat_committed,
                "no_verdict": n_committed - n_pass_committed - n_caveat_committed,
            },
            "hallucination_rate": hallucination_rate,
            "calibrated_abstention_rate": calibrated_abstention,
        },
    )


__all__ = ["measure"]
