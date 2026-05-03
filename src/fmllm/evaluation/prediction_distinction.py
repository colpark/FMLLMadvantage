"""Prediction distinction test (Layer 2).

Hypothesis: non-equivalent queries must produce separable predictions.

The metric is the median across-class final-claim distance over
randomly sampled pairs from different ``(N, motif)`` classes. Higher
is better. Pre-registered threshold ``>= 2.0``.

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
    physical_equivalence_class,
)
from fmllm.orchestrator import Trajectory


def measure(
    *,
    trajectories: list[Trajectory],
    truth: dict[int, dict[str, Any]],
    threshold: float = 2.0,
    n_pairs: int = 200,
    seed: int = 0,
) -> TestResult:
    by_class: dict[tuple[int, str], list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        if t.specimen_id is None or t.specimen_id not in truth:
            continue
        if t.final_claim is None:
            continue
        by_class[physical_equivalence_class(truth[t.specimen_id])].append(t)
    classes = list(by_class.keys())
    if len(classes) < 2:
        return make_skipped(
            test_name="prediction_distinction",
            layer="prediction",
            metric_name="median_across_class_claim_distance",
            threshold=threshold,
            threshold_direction="ge",
            reason=f"only {len(classes)} class(es) available",
        )

    rng = random.Random(seed)
    dists: list[float] = []
    attempts = 0
    while len(dists) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        c1, c2 = rng.sample(classes, 2)
        a = rng.choice(by_class[c1])
        b = rng.choice(by_class[c2])
        d = claim_distance(a.final_claim, b.final_claim)
        if d != float("inf"):
            dists.append(d)

    if not dists:
        return make_skipped(
            test_name="prediction_distinction",
            layer="prediction",
            metric_name="median_across_class_claim_distance",
            threshold=threshold,
            threshold_direction="ge",
            reason="no finite-distance across-class pairs found",
        )

    median = float(statistics.median(dists))
    passes = threshold_check(median, threshold, "ge")
    return TestResult(
        test_name="prediction_distinction",
        layer="prediction",
        metric_name="median_across_class_claim_distance",
        metric_value=median,
        threshold=threshold,
        threshold_direction="ge",
        passes=passes,
        n_samples=len(dists),
        details={
            "n_classes": len(classes),
            "n_pairs_evaluated": len(dists),
        },
    )


__all__ = ["measure"]
