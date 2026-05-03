"""Step recoverability test (Layer 1).

Hypothesis: each verified step's typed claim must agree with the
verifier's source signals at that step.

Operationalization: for every commit / final step in every
trajectory, compare the LLM's claim against the FM-derived
quantities the bridges surfaced in earlier observation steps. The
metric is the fraction of fields whose claimed value falls within a
tolerance of the corresponding FM-derived value.

Comparisons performed when the data is present:

    - claim.n_atoms vs FM1's count head argmax (off-by-N ≤ 2).
    - claim.temperature vs FM3's alpha * beta (relative err ≤ 25%).
    - claim.per_atom_potential_energy vs FM2's value (abs err ≤ 0.5).

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
from fmllm.evaluation.utils import extract_observations
from fmllm.orchestrator import StepType, Trajectory


def measure(
    *,
    trajectories: list[Trajectory],
    threshold: float = 0.70,
    atom_count_tolerance: int = 2,
    temperature_rel_tolerance: float = 0.25,
    energy_abs_tolerance: float = 0.5,
) -> TestResult:
    """Run step recoverability across a list of trajectories."""
    total_checks = 0
    total_pass = 0
    per_field: dict[str, dict[str, int]] = {
        "n_atoms": {"checked": 0, "passed": 0},
        "temperature": {"checked": 0, "passed": 0},
        "per_atom_potential_energy": {"checked": 0, "passed": 0},
    }

    for t in trajectories:
        observations = extract_observations(t)
        for s in t.steps:
            if s.step_type not in (StepType.HYPOTHESIS, StepType.FINAL):
                continue
            claim = s.claim
            if claim is None:
                continue

            # n_atoms vs FM1.
            if claim.n_atoms is not None and "fm1_image" in observations:
                fm1_value = observations["fm1_image"].prediction.value or {}
                n_pred = fm1_value.get("n_atoms_pred")
                if n_pred is not None:
                    per_field["n_atoms"]["checked"] += 1
                    total_checks += 1
                    if abs(int(claim.n_atoms) - int(n_pred)) <= atom_count_tolerance:
                        per_field["n_atoms"]["passed"] += 1
                        total_pass += 1

            # temperature vs FM3.
            if claim.temperature is not None and "fm3_traj" in observations:
                fm3_value = observations["fm3_traj"].prediction.value or {}
                alpha = fm3_value.get("alpha")
                beta = fm3_value.get("beta")
                if alpha is not None and beta is not None:
                    pred_t = float(alpha) * float(beta)
                    per_field["temperature"]["checked"] += 1
                    total_checks += 1
                    denom = max(abs(claim.temperature), 1.0e-3)
                    if abs(pred_t - claim.temperature) / denom <= temperature_rel_tolerance:
                        per_field["temperature"]["passed"] += 1
                        total_pass += 1

            # per_atom_potential_energy vs FM2.
            if (
                claim.per_atom_potential_energy is not None
                and "fm2_rdf" in observations
            ):
                fm2_value = observations["fm2_rdf"].prediction.value or {}
                pred_e = fm2_value.get("value_lj")
                if pred_e is not None:
                    per_field["per_atom_potential_energy"]["checked"] += 1
                    total_checks += 1
                    if abs(float(pred_e) - claim.per_atom_potential_energy) <= energy_abs_tolerance:
                        per_field["per_atom_potential_energy"]["passed"] += 1
                        total_pass += 1

    if total_checks == 0:
        return make_skipped(
            test_name="step_recoverability",
            layer="trajectory",
            metric_name="frac_recoverable",
            threshold=threshold,
            threshold_direction="ge",
            reason="no commit/hypothesis steps with FM observations available",
        )

    frac = total_pass / total_checks
    passes = threshold_check(frac, threshold, "ge")
    return TestResult(
        test_name="step_recoverability",
        layer="trajectory",
        metric_name="frac_recoverable",
        metric_value=frac,
        threshold=threshold,
        threshold_direction="ge",
        passes=passes,
        n_samples=total_checks,
        details={
            "total_checks": total_checks,
            "total_pass": total_pass,
            "per_field": per_field,
        },
    )


__all__ = ["measure"]
