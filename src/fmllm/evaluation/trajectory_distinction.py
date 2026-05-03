"""Trajectory distinction test (Layer 1).

Hypothesis: pairs of trajectories arriving at non-equivalent states
must produce separable trajectories.

Operationalization: pick pairs of trajectories from different
``(N, motif)`` equivalence classes. Compute action-signature and
claim distances. The metric is the median across-class distance.

Higher is better. The pre-registered threshold demands median
across-class claim distance to exceed twice the within-class median
(equivalently, at least ``2.0`` LJ-equivalent units in our metric).

Depends on:
    fmllm.evaluation.utils, fmllm.evaluation.schema.
"""

from __future__ import annotations

import random
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
    action_threshold: float = 1.0,
    claim_threshold: float = 2.0,
    n_pairs: int = 200,
    seed: int = 0,
    only_passing: bool = False,
) -> TestResult:
    """Run the trajectory-distinction test.

    Samples up to ``n_pairs`` random across-class trajectory pairs and
    computes median action / claim distances.
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
    classes = list(by_class.keys())
    if len(classes) < 2:
        return make_skipped(
            test_name="trajectory_distinction",
            layer="trajectory",
            metric_name="median_across_class_distance",
            threshold=action_threshold,
            threshold_direction="ge",
            reason=f"only {len(classes)} equivalence class(es) available",
        )

    rng = random.Random(seed)
    action_dists: list[float] = []
    claim_dists: list[float] = []
    attempts = 0
    while len(action_dists) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        c1, c2 = rng.sample(classes, 2)
        a = rng.choice(by_class[c1])
        b = rng.choice(by_class[c2])
        action_dists.append(
            float(edit_distance(
                trajectory_action_signature(a),
                trajectory_action_signature(b),
            ))
        )
        claim_dists.append(claim_distance(a.final_claim, b.final_claim))

    median_action = float(statistics.median(action_dists))
    finite_claims = [d for d in claim_dists if d != float("inf")]
    median_claim = float(statistics.median(finite_claims)) if finite_claims else 0.0

    passes = (
        threshold_check(median_action, action_threshold, "ge")
        and threshold_check(median_claim, claim_threshold, "ge")
    )
    return TestResult(
        test_name="trajectory_distinction",
        layer="trajectory",
        metric_name="median_across_class_action_distance",
        metric_value=median_action,
        threshold=action_threshold,
        threshold_direction="ge",
        passes=passes,
        n_samples=len(action_dists),
        details={
            "median_across_class_action_distance": median_action,
            "median_across_class_claim_distance": median_claim,
            "claim_threshold": claim_threshold,
            "n_classes": len(classes),
            "n_pairs_evaluated": len(action_dists),
            "only_passing": only_passing,
        },
    )


__all__ = ["measure"]
