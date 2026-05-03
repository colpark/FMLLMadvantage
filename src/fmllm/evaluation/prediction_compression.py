"""Prediction compression test (Layer 2).

Hypothesis: equivalent queries (paraphrases, equivalent measurement
modalities) must produce predictions whose structural components
cluster tightly.

In this testbed we don't run paraphrased queries; "equivalent" maps
to specimens sharing the same ``(N, motif)``. Within a class,
temperature varies legitimately by construction (it is NOT part of
the equivalence relation), so the metric uses
:func:`fmllm.evaluation.utils.claim_distance_structural` which
excludes temperature and energy. The metric itself is the median
within-class mean pairwise structural claim distance.

Lower is better. Pre-registered threshold: median within-class
structural claim distance ``<= 1.0``.

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
    claim_distance_structural,
    physical_equivalence_class,
)
from fmllm.orchestrator import Trajectory


def measure(
    *,
    trajectories: list[Trajectory],
    truth: dict[int, dict[str, Any]],
    threshold: float = 1.0,
) -> TestResult:
    by_class: dict[tuple[int, str], list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        if t.specimen_id is None or t.specimen_id not in truth:
            continue
        if t.final_claim is None:
            continue
        by_class[physical_equivalence_class(truth[t.specimen_id])].append(t)

    class_means: list[float] = []
    n_pairs = 0
    for cls, members in by_class.items():
        if len(members) < 2:
            continue
        ds: list[float] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                ds.append(
                    claim_distance_structural(
                        members[i].final_claim, members[j].final_claim,
                    )
                )
                n_pairs += 1
        finite = [d for d in ds if d != float("inf")]
        if finite:
            class_means.append(statistics.mean(finite))

    if not class_means:
        return make_skipped(
            test_name="prediction_compression",
            layer="prediction",
            metric_name="median_within_class_structural_claim_distance",
            threshold=threshold,
            threshold_direction="le",
            reason="no equivalence class with 2+ committed-claim trajectories",
        )

    median_within = float(statistics.median(class_means))
    passes = threshold_check(median_within, threshold, "le")
    return TestResult(
        test_name="prediction_compression",
        layer="prediction",
        metric_name="median_within_class_structural_claim_distance",
        metric_value=median_within,
        threshold=threshold,
        threshold_direction="le",
        passes=passes,
        n_samples=n_pairs,
        details={
            "claim_metric": "structural",
            "n_classes_with_pairs": len(class_means),
            "n_pairs": n_pairs,
        },
    )


__all__ = ["measure"]
