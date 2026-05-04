"""Trajectory distinction test (Layer 1).

Hypothesis: pairs of trajectories arriving at non-equivalent states
must produce separable trajectories.

Operationalization: pick pairs of trajectories from different
``(N, motif)`` equivalence classes. The headline metric is the
median across-class **claim** distance, which captures whether the
final commit reflects the input class. Action-signature distance is
recorded in the details but does not gate the pass/fail flag: a
well-behaved orchestrator follows the same evidence-gathering
protocol on every specimen (call fm1, call fm2, call fm3, commit),
so action distance is structurally near zero. Asking the actions to
*also* differ across classes would penalize exactly the consistent
protocol we want, so this test gates on claim distance only.

Higher is better. The pre-registered threshold demands median
across-class claim distance ``>= 2.0`` (the typical N + motif gap
between distinct classes is well above this).

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
    threshold: float = 2.0,
    n_pairs: int = 200,
    seed: int = 0,
    only_passing: bool = False,
) -> TestResult:
    """Run the trajectory-distinction test.

    The metric is the median across-class claim distance over
    randomly sampled pairs from different ``(N, motif)`` classes.
    Action distance is recorded informationally in the details but
    does not gate pass/fail.
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
            metric_name="median_across_class_claim_distance",
            threshold=threshold,
            threshold_direction="ge",
            reason=f"only {len(classes)} equivalence class(es) available",
        )

    rng = random.Random(seed)
    action_dists: list[float] = []
    claim_dists: list[float] = []
    attempts = 0
    while len(claim_dists) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        c1, c2 = rng.sample(classes, 2)
        a = rng.choice(by_class[c1])
        b = rng.choice(by_class[c2])
        d_claim = claim_distance(a.final_claim, b.final_claim)
        if d_claim == float("inf"):
            continue
        action_dists.append(
            float(edit_distance(
                trajectory_action_signature(a),
                trajectory_action_signature(b),
            ))
        )
        claim_dists.append(d_claim)

    if not claim_dists:
        return make_skipped(
            test_name="trajectory_distinction",
            layer="trajectory",
            metric_name="median_across_class_claim_distance",
            threshold=threshold,
            threshold_direction="ge",
            reason="no finite-distance across-class commit pairs found",
        )

    median_claim = float(statistics.median(claim_dists))
    median_action = (
        float(statistics.median(action_dists)) if action_dists else 0.0
    )

    passes = threshold_check(median_claim, threshold, "ge")
    return TestResult(
        test_name="trajectory_distinction",
        layer="trajectory",
        metric_name="median_across_class_claim_distance",
        metric_value=median_claim,
        threshold=threshold,
        threshold_direction="ge",
        passes=passes,
        n_samples=len(claim_dists),
        details={
            "median_across_class_claim_distance": median_claim,
            "median_across_class_action_distance": median_action,
            "n_classes": len(classes),
            "n_pairs_evaluated": len(claim_dists),
            "only_passing": only_passing,
            "note": (
                "action distance is informational; gating on it would "
                "penalize systems with consistent evidence-gathering "
                "protocols. claim distance is the headline metric."
            ),
        },
    )


__all__ = ["measure"]
