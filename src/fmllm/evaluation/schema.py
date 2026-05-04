"""Pydantic schemas for evaluation outputs.

Each of the eight world-model tests produces a :class:`TestResult`
with a metric value, the threshold that defines pass/fail, the
boolean outcome, and free-form details. The harness aggregates the
eight results into an :class:`EvaluationReport` that serializes to
YAML for downstream analysis and Phase 8 experiments.

Depends on:
    pydantic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TestResult(_StrictModel):
    """One world-model test's outcome."""

    test_name: str
    layer: str  # "trajectory", "prediction", or "cross_layer"
    metric_name: str
    metric_value: float | None = None
    threshold: float | None = None
    threshold_direction: str = "ge"  # "ge" | "le" | "eq"
    passes: bool
    n_samples: int = 0
    details: dict[str, Any] = Field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""


class EvaluationReport(_StrictModel):
    """Aggregate of the eight world-model tests plus optional
    ground-truth accuracy."""

    run_id: str
    timestamp_utc: str
    trajectory_results: list[TestResult] = Field(default_factory=list)
    prediction_results: list[TestResult] = Field(default_factory=list)
    cross_layer_results: list[TestResult] = Field(default_factory=list)
    accuracy_results: list[TestResult] = Field(default_factory=list)
    aggregate_pass: bool = False
    inputs: dict[str, Any] = Field(default_factory=dict)


def make_skipped(
    *,
    test_name: str,
    layer: str,
    metric_name: str,
    threshold: float,
    threshold_direction: str = "ge",
    reason: str = "no data",
) -> TestResult:
    return TestResult(
        test_name=test_name,
        layer=layer,
        metric_name=metric_name,
        metric_value=None,
        threshold=threshold,
        threshold_direction=threshold_direction,
        passes=False,
        n_samples=0,
        skipped=True,
        skip_reason=reason,
    )


def threshold_check(
    metric: float, threshold: float, direction: str = "ge",
) -> bool:
    """Return whether ``metric`` satisfies the threshold."""
    if direction == "ge":
        return metric >= threshold
    if direction == "le":
        return metric <= threshold
    if direction == "eq":
        return abs(metric - threshold) <= 0.05
    raise ValueError(f"unknown threshold direction {direction!r}")


__all__ = [
    "EvaluationReport",
    "TestResult",
    "make_skipped",
    "threshold_check",
]
