"""Probe: FM2 non-negativity relative to the LJ energy floor.

For each test specimen, the probe checks whether the predicted
per-atom energy stays at or above the configured floor. Score is the
fraction of specimens whose prediction respects the floor.
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
    n_samples = int(config.get("n_samples", min(128, len(items))))
    floor = float(config.get("energy_floor", -3.0))

    if not items:
        return ProbeResult(
            constraint_name="non_negativity",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_predictions_above_floor",
            passes_threshold=False, threshold=threshold,
            details={"reason": "no items provided"},
        )

    model.eval()
    rdfs = torch.stack(
        [items[i]["rdf"] for i in range(min(n_samples, len(items)))],
        dim=0,
    ).to(device)
    with torch.no_grad():
        preds = model(rdfs).cpu()
    above = (preds >= floor).int()
    matched = int(above.sum().item())
    total = int(above.numel())
    score = matched / max(1, total)
    return ProbeResult(
        constraint_name="non_negativity",
        satisfaction_score=score,
        num_test_cases=total,
        metric="frac_predictions_above_floor",
        passes_threshold=score >= threshold,
        threshold=threshold,
        details={"floor": floor, "matched": matched, "min_pred": float(preds.min())},
    )


__all__ = ["run_probe"]
