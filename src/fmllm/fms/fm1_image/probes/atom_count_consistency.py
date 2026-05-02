"""Probe: FM1 count-head and confidence-head agree on atom count.

For each test specimen, the count head's argmax should match the
number of query slots whose objectness logit exceeds zero (i.e.,
sigmoid(logit) > 0.5). Score is the fraction of specimens where the
two estimates agree exactly.
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
    threshold = float(config.get("threshold", 0.95))
    n_samples = int(config.get("n_samples", min(64, len(items))))

    if not items:
        return ProbeResult(
            constraint_name="atom_count_consistency",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_count_matches_thresholded_queries",
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
    pred_count = out["count_logits"].argmax(dim=-1).cpu()
    confs = torch.sigmoid(out["confidence_logits"]).cpu()
    n_above = (confs > 0.5).sum(dim=-1)
    matches = (pred_count == n_above).int()
    matched = int(matches.sum().item())
    total = int(matches.numel())
    score = matched / max(1, total)
    return ProbeResult(
        constraint_name="atom_count_consistency",
        satisfaction_score=score,
        num_test_cases=total,
        metric="frac_count_matches_thresholded_queries",
        passes_threshold=score >= threshold,
        threshold=threshold,
        details={"matched": matched, "mean_count_pred": float(pred_count.float().mean())},
    )


__all__ = ["run_probe"]
