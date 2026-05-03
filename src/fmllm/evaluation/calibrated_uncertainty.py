"""Calibrated uncertainty test (cross-layer).

Hypothesis: each FM's calibrated prediction band covers ground truth
at the claimed level. The training pipeline calibrates split-conformal
thresholds at ``alpha = 0.10`` (confidence level ``0.90``). Empirical
band-membership on the held-out trajectories should match within
tolerance.

This test reads conformal verdicts from every step's verifier output
in committed trajectories. The conformal source emits, per FM, a
``flag`` field with values:

    "ok"                       FM in-distribution and (when checked) claim inside band
    "out_of_distribution"      FM raised the OOD flag
    "claim_outside_band"       claim's value sat outside the calibrated band

The metric is the absolute gap between empirical clean rate per FM
(``"ok"`` / total verdicts) and the claimed coverage ``1 - alpha``,
averaged across FMs. Lower is better. Pre-registered threshold ``<= 0.10``.

The default claimed coverage matches the calibrate step's default; an
explicit ``claimed_coverage`` argument overrides it.

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


def _iter_per_fm_flags(
    trajectories: list[Trajectory],
) -> list[tuple[str, str]]:
    """Yield (fm_name, flag) for every conformal per-FM entry across all
    steps in the trajectories."""
    out: list[tuple[str, str]] = []
    for t in trajectories:
        for step in t.steps:
            verdict = getattr(step, "verdict", None)
            if verdict is None:
                continue
            for sv in getattr(verdict, "source_verdicts", []) or []:
                if sv.source_name != "conformal":
                    continue
                evidence = sv.evidence or {}
                per_fm = evidence.get("per_fm") or []
                for entry in per_fm:
                    fm = entry.get("fm_name")
                    flag = entry.get("flag")
                    if fm is None or flag is None:
                        continue
                    out.append((str(fm), str(flag)))
    return out


def measure(
    *,
    trajectories: list[Trajectory],
    threshold: float = 0.10,
    claimed_coverage: float = 0.90,
) -> TestResult:
    flags = _iter_per_fm_flags(trajectories)
    if not flags:
        return make_skipped(
            test_name="calibrated_uncertainty",
            layer="cross_layer",
            metric_name="mean_abs_coverage_gap",
            threshold=threshold,
            threshold_direction="le",
            reason="no conformal verdicts found in trajectories",
        )

    by_fm: dict[str, list[str]] = defaultdict(list)
    for fm, flag in flags:
        by_fm[fm].append(flag)

    per_fm: list[dict[str, Any]] = []
    gaps: list[float] = []
    for fm, fm_flags in by_fm.items():
        n = len(fm_flags)
        n_ok = sum(1 for f in fm_flags if f == "ok")
        empirical = n_ok / n if n else 0.0
        gap = abs(empirical - claimed_coverage)
        gaps.append(gap)
        per_fm.append(
            {
                "fm": fm,
                "n_observations": n,
                "n_ok": n_ok,
                "empirical_coverage": empirical,
                "claimed_coverage": claimed_coverage,
                "abs_gap": gap,
            }
        )

    mean_gap = sum(gaps) / len(gaps)
    passes = threshold_check(mean_gap, threshold, "le")
    return TestResult(
        test_name="calibrated_uncertainty",
        layer="cross_layer",
        metric_name="mean_abs_coverage_gap",
        metric_value=mean_gap,
        threshold=threshold,
        threshold_direction="le",
        passes=passes,
        n_samples=len(flags),
        details={
            "claimed_coverage": claimed_coverage,
            "n_fms": len(per_fm),
            "per_fm": per_fm,
        },
    )


__all__ = ["measure"]
