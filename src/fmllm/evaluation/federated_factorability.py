"""Federated factorability test (cross-layer).

Hypothesis: removing or ablating one architectural component must
produce a recoverable, localized drop in capability rather than a
diffuse system-wide collapse. A truly factored system has identifiable
parts; a brittle entangled system fails everywhere when you touch
anything.

In this testbed we compare verifier-pass rates across the
ablation lattice ``V0..V4`` (defined in
:class:`fmllm.verifier.SourcesConfig`):

    V0  rule_library only
    V1  rule_library + literature
    V2  V1 + cross_fm
    V3  V2 + simulator
    V4  full (V3 + conformal)

Localized degradation is captured by two quantities:

* **Monotonicity**: pass-rate increases as more sources come online.
  We compute the fraction of adjacent transitions that improve.
* **Step factor**: ratio of the largest single-step gain to the total
  V0->V4 gain. A factorized system has a small step factor (each
  source contributes some signal). A brittle system has step factor
  near 1 (one source carries everything).

The test passes when monotonicity ``>= 0.75`` AND step_factor
``<= 0.85``. The reported metric is the unweighted average of
``monotonicity`` and ``(1 - step_factor)``; threshold ``>= 0.45``.

Depends on:
    fmllm.evaluation.schema, fmllm.orchestrator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fmllm.evaluation.schema import (
    TestResult,
    make_skipped,
    threshold_check,
)
from fmllm.orchestrator import Trajectory


_ABLATION_ORDER = ["V0", "V1", "V2", "V3", "V4"]


def _pass_rate(trajs: list[Trajectory]) -> float:
    if not trajs:
        return 0.0
    n_pass = sum(
        1 for t in trajs
        if t.final_verdict is not None
        and t.final_verdict.aggregate_decision.value == "pass"
    )
    return n_pass / len(trajs)


def measure(
    *,
    trajectories_by_ablation: dict[str, list[Trajectory]],
    threshold: float = 0.45,
    monotonicity_floor: float = 0.75,
    step_factor_ceiling: float = 0.85,
) -> TestResult:
    """Measure federated factorability across ablations V0..V4.

    ``trajectories_by_ablation`` maps the preset key (e.g. ``"V2"``)
    to the trajectories collected under that ablation.
    """
    present = [a for a in _ABLATION_ORDER if a in trajectories_by_ablation]
    if len(present) < 2:
        return make_skipped(
            test_name="federated_factorability",
            layer="cross_layer",
            metric_name="factorability_score",
            threshold=threshold,
            threshold_direction="ge",
            reason=f"need >=2 ablations, got {present}",
        )

    rates = {a: _pass_rate(trajectories_by_ablation[a]) for a in present}
    counts = {a: len(trajectories_by_ablation[a]) for a in present}

    transitions = []
    for i in range(len(present) - 1):
        a, b = present[i], present[i + 1]
        transitions.append((a, b, rates[b] - rates[a]))

    n_improving = sum(1 for *_, d in transitions if d > 0.0)
    monotonicity = n_improving / max(1, len(transitions))

    total_gain = rates[present[-1]] - rates[present[0]]
    if total_gain <= 0.0:
        step_factor = 1.0
    else:
        max_step = max((d for *_, d in transitions if d > 0.0), default=0.0)
        step_factor = max_step / total_gain

    score = 0.5 * monotonicity + 0.5 * (1.0 - step_factor)
    passes_threshold = threshold_check(score, threshold, "ge")
    structural_ok = (
        monotonicity >= monotonicity_floor
        and step_factor <= step_factor_ceiling
    )
    passes = passes_threshold and structural_ok

    return TestResult(
        test_name="federated_factorability",
        layer="cross_layer",
        metric_name="factorability_score",
        metric_value=score,
        threshold=threshold,
        threshold_direction="ge",
        passes=passes,
        n_samples=sum(counts.values()),
        details={
            "ablations": present,
            "pass_rates": rates,
            "counts": counts,
            "transitions": [
                {"from": a, "to": b, "gain": d} for a, b, d in transitions
            ],
            "monotonicity": monotonicity,
            "monotonicity_floor": monotonicity_floor,
            "step_factor": step_factor,
            "step_factor_ceiling": step_factor_ceiling,
            "structural_ok": structural_ok,
        },
    )


__all__ = ["measure"]
