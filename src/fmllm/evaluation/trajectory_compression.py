"""Trajectory compression test (Layer 1).

Hypothesis: pairs of trajectories converging to equivalent verifier-
signed states must produce equivalent intermediate-state sequences
and equivalent final predictions.

Operationalization: group trajectories into physical-equivalence
classes by ``(N, motif)``. Within each class, compute pairwise
distances between (a) action signatures, and (b) final claims.
The metric is the median of the within-class median distances.

Lower is better. The pre-registered threshold demands median within-
class action-signature distance ``<= 2`` (at most a couple of edits)
and median within-class claim distance ``<= 1.0`` (close in atom
count, similar T, similar motif).

Depends on:
    fmllm.evaluation.utils, fmllm.evaluation.schema.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from fmllm.evaluation.schema import (
    TestResult,
    make_skipped,
    threshold_check,
)
from fmllm.evaluation.utils import (
    claim_distance,
    edit_distance,
    physical_equivalence_class,
    trajectory_action_signature,
)
from fmllm.orchestrator import Trajectory


def measure(
    *,
    trajectories: list[Trajectory],
    truth: dict[int, dict[str, Any]],
    action_threshold: float = 2.0,
    claim_threshold: float = 1.0,
    only_passing: bool = False,
) -> TestResult:
    """Run the trajectory-compression test.

    Args:
        trajectories: list of trajectories from a single pipeline run.
        truth: per-specimen ground truth (from
            ``utils.load_ground_truth``).
        action_threshold: max acceptable median within-class
            action-signature edit distance.
        claim_threshold: max acceptable median within-class claim
            distance.
        only_passing: when True, restrict to verifier-PASS trajectories.
            CAVEAT and FAIL trajectories drop out.
    """
    if only_passing:
        trajectories = [
            t for t in trajectories
            if t.final_verdict is not None
            and t.final_verdict.aggregate_decision.value == "pass"
        ]

    by_class: dict[tuple[int, str], list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        if t.specimen_id is None or t.specimen_id not in truth:
            continue
        by_class[physical_equivalence_class(truth[t.specimen_id])].append(t)

    action_medians: list[float] = []
    claim_medians: list[float] = []
    n_pairs = 0
    for cls, members in by_class.items():
        if len(members) < 2:
            continue
        action_dists: list[float] = []
        claim_dists: list[float] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a = members[i]
                b = members[j]
                action_dists.append(
                    float(edit_distance(
                        trajectory_action_signature(a),
                        trajectory_action_signature(b),
                    ))
                )
                claim_dists.append(claim_distance(a.final_claim, b.final_claim))
                n_pairs += 1
        if action_dists:
            action_medians.append(statistics.median(action_dists))
        if claim_dists:
            claim_medians.append(statistics.median(claim_dists))

    if not action_medians:
        return make_skipped(
            test_name="trajectory_compression",
            layer="trajectory",
            metric_name="median_within_class_distance",
            threshold=action_threshold,
            threshold_direction="le",
            reason=(
                "no equivalence class has 2+ trajectories "
                f"(checked {len(by_class)} classes)"
            ),
        )

    median_action = float(statistics.median(action_medians))
    median_claim = (
        float(statistics.median(claim_medians)) if claim_medians else float("inf")
    )

    passes = (
        threshold_check(median_action, action_threshold, "le")
        and threshold_check(median_claim, claim_threshold, "le")
    )
    return TestResult(
        test_name="trajectory_compression",
        layer="trajectory",
        metric_name="median_within_class_action_distance",
        metric_value=median_action,
        threshold=action_threshold,
        threshold_direction="le",
        passes=passes,
        n_samples=n_pairs,
        details={
            "median_within_class_action_distance": median_action,
            "median_within_class_claim_distance": median_claim,
            "claim_threshold": claim_threshold,
            "n_classes_with_pairs": len(action_medians),
            "n_pairs": n_pairs,
            "only_passing": only_passing,
        },
    )


__all__ = ["measure"]
