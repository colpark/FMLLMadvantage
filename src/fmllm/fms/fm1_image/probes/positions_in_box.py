"""Probe: FM1 predicted positions stay inside the imaging box.

For each test specimen, the probe checks every confident query slot
(objectness sigmoid > 0.5) and counts how many predicted positions
lie inside ``[-box_half_width_lj, box_half_width_lj]`` along both
axes. Score is the fraction of confident-and-in-box atoms.
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
    threshold = float(config.get("threshold", 0.99))
    n_samples = int(config.get("n_samples", min(64, len(items))))
    box_half_width_lj = float(config.get("box_half_width_lj", 4.8))

    if not items:
        return ProbeResult(
            constraint_name="positions_in_box",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_confident_atoms_in_box",
            passes_threshold=False, threshold=threshold,
            details={"reason": "no items provided"},
        )

    model.eval()
    images = torch.stack(
        [items[i]["image"] for i in range(min(n_samples, len(items)))],
        dim=0,
    ).to(device)
    with torch.no_grad():
        out = model(images)
    confs = torch.sigmoid(out["confidence_logits"]).cpu()
    positions = out["positions"].cpu()

    in_box = (positions.abs() <= box_half_width_lj).all(dim=-1)  # (B, Q)
    confident = confs > 0.5
    confident_count = int(confident.sum().item())
    if confident_count == 0:
        return ProbeResult(
            constraint_name="positions_in_box",
            satisfaction_score=1.0,
            num_test_cases=int(positions.shape[0]),
            metric="frac_confident_atoms_in_box",
            passes_threshold=True,
            threshold=threshold,
            details={"reason": "no confident queries; vacuous score"},
        )
    in_box_and_confident = int((in_box & confident).sum().item())
    score = in_box_and_confident / confident_count
    return ProbeResult(
        constraint_name="positions_in_box",
        satisfaction_score=score,
        num_test_cases=int(positions.shape[0]),
        metric="frac_confident_atoms_in_box",
        passes_threshold=score >= threshold,
        threshold=threshold,
        details={
            "confident_count": confident_count,
            "in_box_count": in_box_and_confident,
            "box_half_width_lj": box_half_width_lj,
        },
    )


__all__ = ["run_probe"]
