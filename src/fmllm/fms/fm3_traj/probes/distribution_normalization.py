"""Probe: FM3 Gamma distribution normalization.

The Gamma probability density integrates to one over [0, infinity)
analytically. The probe verifies this numerically with a coarse
trapezoidal quadrature over a wide range determined by the predicted
moments. Score is the fraction of specimens whose numerical integral
lies within ``rel_tolerance`` of unity.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Gamma

from fmllm.fms._schemas import ProbeResult


def run_probe(
    *,
    model: nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
) -> ProbeResult:
    threshold = float(config.get("threshold", 1.0))
    n_samples = int(config.get("n_samples", min(32, len(items))))
    rel_tolerance = float(config.get("rel_tolerance", 0.05))
    grid_points = int(config.get("grid_points", 4096))

    if not items:
        return ProbeResult(
            constraint_name="distribution_normalization",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_integral_close_to_one",
            passes_threshold=False, threshold=threshold,
            details={"reason": "no items provided"},
        )

    model.eval()
    chosen = items[: min(n_samples, len(items))]
    traj_pos = torch.stack([it["traj_positions"] for it in chosen], dim=0).to(device)
    traj_vel = torch.stack([it["traj_velocities"] for it in chosen], dim=0).to(device)
    atom_mask = torch.stack([it["atom_mask"] for it in chosen], dim=0).to(device)

    with torch.no_grad():
        out = model(traj_pos, traj_vel, atom_mask)
    alpha = out["alpha"].cpu()
    beta = out["beta"].cpu()

    matched = 0
    integrals = []
    for a, b in zip(alpha.tolist(), beta.tolist(), strict=True):
        max_x = max(20.0 * float(a) * float(b), 50.0)
        x = torch.linspace(1.0e-6, max_x, grid_points)
        rate = 1.0 / max(b, 1.0e-8)
        log_prob = Gamma(torch.tensor(a), torch.tensor(rate)).log_prob(x)
        density = torch.exp(log_prob)
        integral = float(torch.trapz(density, x))
        integrals.append(integral)
        if abs(integral - 1.0) <= rel_tolerance:
            matched += 1
    total = len(integrals)
    score = matched / max(1, total)
    return ProbeResult(
        constraint_name="distribution_normalization",
        satisfaction_score=score,
        num_test_cases=total,
        metric="frac_integral_close_to_one",
        passes_threshold=score >= threshold,
        threshold=threshold,
        details={
            "rel_tolerance": rel_tolerance,
            "mean_integral": sum(integrals) / max(1, len(integrals)),
        },
    )


__all__ = ["run_probe"]
