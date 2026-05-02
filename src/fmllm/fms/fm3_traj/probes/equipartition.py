"""Probe: FM3 equipartition (alpha * beta vs empirical mean KE).

For each test specimen, the probe runs the model on the trajectory
snippet, reads ``alpha * beta`` from the predicted Gamma moments,
computes the empirical per-atom KE from the recorded velocities, and
checks whether ``|alpha * beta - mean_KE| / mean_KE`` falls below the
configured tolerance.
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
    threshold = float(config.get("threshold", 0.85))
    n_samples = int(config.get("n_samples", min(64, len(items))))
    rel_tolerance = float(config.get("rel_tolerance", 0.20))

    if not items:
        return ProbeResult(
            constraint_name="equipartition",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_within_rel_tol",
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
    pred_mean = (out["alpha"] * out["beta"]).cpu()

    ke = 0.5 * (traj_vel * traj_vel).sum(dim=-1)  # (B, T, N)
    mask = atom_mask.unsqueeze(1).expand_as(ke)
    ke_zero = ke.masked_fill(~mask, 0.0)
    n_real = mask.sum(dim=(-1, -2)).clamp(min=1).to(ke.dtype)
    obs_mean = (ke_zero.sum(dim=(-1, -2)) / n_real).cpu()

    denom = obs_mean.abs().clamp(min=1.0e-6)
    rel_err = ((pred_mean - obs_mean).abs() / denom)
    matched = int((rel_err <= rel_tolerance).sum().item())
    total = int(rel_err.numel())
    score = matched / max(1, total)
    return ProbeResult(
        constraint_name="equipartition",
        satisfaction_score=score,
        num_test_cases=total,
        metric="frac_within_rel_tol",
        passes_threshold=score >= threshold,
        threshold=threshold,
        details={
            "rel_tolerance": rel_tolerance,
            "matched": matched,
            "mean_pred_alpha_beta": float(pred_mean.mean()),
            "mean_obs_ke": float(obs_mean.mean()),
        },
    )


__all__ = ["run_probe"]
