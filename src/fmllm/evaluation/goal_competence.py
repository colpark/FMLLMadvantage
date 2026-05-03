"""Goal competence test (Layer 2).

Hypothesis: failure rate on held-out compound goals up to depth ``n``
remains bounded.

In this testbed a compound goal is a conjunction of simple
conditions on the typed claim. We evaluate three depth-1 goals
(""is the cluster size correct?", "is the temperature within 25%?",
"is the motif correct?") and one depth-2 goal (size AND motif).

The metric is the failure rate across goals and specimens. Lower is
better. Pre-registered threshold: failure rate ``<= 0.50``.

Depends on:
    fmllm.evaluation.utils, fmllm.evaluation.schema.
"""

from __future__ import annotations

from typing import Any

from fmllm.evaluation.schema import (
    TestResult,
    make_skipped,
    threshold_check,
)
from fmllm.orchestrator import Trajectory


def _goal_size(claim, truth) -> bool | None:
    if claim is None or claim.n_atoms is None:
        return None
    return abs(int(claim.n_atoms) - int(truth["n"])) <= 2


def _goal_temperature(claim, truth) -> bool | None:
    if claim is None or claim.temperature is None:
        return None
    denom = max(abs(truth["t"]), 1.0e-3)
    return abs(claim.temperature - truth["t"]) / denom <= 0.25


def _goal_motif(claim, truth) -> bool | None:
    if claim is None or claim.motif is None:
        return None
    return claim.motif == truth["motif"]


def measure(
    *,
    trajectories: list[Trajectory],
    truth: dict[int, dict[str, Any]],
    threshold: float = 0.50,
) -> TestResult:
    """Evaluate goal competence across simple and compound goals."""
    goal_results = {
        "size": {"checked": 0, "passed": 0},
        "temperature": {"checked": 0, "passed": 0},
        "motif": {"checked": 0, "passed": 0},
        "size_and_motif": {"checked": 0, "passed": 0},
    }

    for t in trajectories:
        if t.specimen_id is None or t.specimen_id not in truth:
            continue
        gt = truth[t.specimen_id]
        claim = t.final_claim

        size_ok = _goal_size(claim, gt)
        temp_ok = _goal_temperature(claim, gt)
        motif_ok = _goal_motif(claim, gt)

        if size_ok is not None:
            goal_results["size"]["checked"] += 1
            if size_ok:
                goal_results["size"]["passed"] += 1
        if temp_ok is not None:
            goal_results["temperature"]["checked"] += 1
            if temp_ok:
                goal_results["temperature"]["passed"] += 1
        if motif_ok is not None:
            goal_results["motif"]["checked"] += 1
            if motif_ok:
                goal_results["motif"]["passed"] += 1
        if size_ok is not None and motif_ok is not None:
            goal_results["size_and_motif"]["checked"] += 1
            if size_ok and motif_ok:
                goal_results["size_and_motif"]["passed"] += 1

    total_checked = sum(g["checked"] for g in goal_results.values())
    total_passed = sum(g["passed"] for g in goal_results.values())

    if total_checked == 0:
        return make_skipped(
            test_name="goal_competence",
            layer="prediction",
            metric_name="failure_rate",
            threshold=threshold,
            threshold_direction="le",
            reason="no committed claims with comparable fields",
        )

    failure_rate = 1.0 - (total_passed / total_checked)
    passes = threshold_check(failure_rate, threshold, "le")
    per_goal_failure = {
        name: (1.0 - g["passed"] / g["checked"]) if g["checked"] else None
        for name, g in goal_results.items()
    }
    return TestResult(
        test_name="goal_competence",
        layer="prediction",
        metric_name="failure_rate",
        metric_value=failure_rate,
        threshold=threshold,
        threshold_direction="le",
        passes=passes,
        n_samples=total_checked,
        details={
            "per_goal_failure_rate": per_goal_failure,
            "per_goal_counts": goal_results,
        },
    )


__all__ = ["measure"]
