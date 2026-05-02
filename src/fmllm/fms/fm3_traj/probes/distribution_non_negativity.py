"""Probe: FM3 Gamma parameters stay strictly positive.

The model uses ``softplus(raw) + 1e-3`` to enforce positive ``alpha``
and ``beta``. The probe confirms this in practice on the test set.
Score is the fraction of specimens with both parameters strictly
positive.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fmllm.fms._schemas import ProbeResult


def run_probe(
    *,
    model: nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
) -> ProbeResult:
    threshold = float(config.get("threshold", 1.0))
    n_samples = int(config.get("n_samples", min(64, len(items))))

    if not items:
        return ProbeResult(
            constraint_name="distribution_non_negativity",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_alpha_beta_positive",
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
    positive = ((alpha > 0) & (beta > 0)).int()
    matched = int(positive.sum().item())
    total = int(positive.numel())
    score = matched / max(1, total)
    return ProbeResult(
        constraint_name="distribution_non_negativity",
        satisfaction_score=score,
        num_test_cases=total,
        metric="frac_alpha_beta_positive",
        passes_threshold=score >= threshold,
        threshold=threshold,
        details={
            "min_alpha": float(alpha.min()),
            "min_beta": float(beta.min()),
        },
    )


__all__ = ["run_probe"]
