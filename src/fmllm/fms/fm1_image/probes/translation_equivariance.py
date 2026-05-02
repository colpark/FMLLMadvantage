"""Probe: FM1 translation equivariance.

Conv-based patch embedding gives exact equivariance to image shifts
that are integer multiples of ``patch_size`` pixels. The probe rolls
each test image by ``patch_size`` along both axes and measures whether
the predicted positions shift by the same amount in LJ units. Score is
the fraction of matched query slots whose shift error stays below the
configured pixel-tolerance.

Depends on:
    torch, scipy.optimize.linear_sum_assignment, numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from fmllm.fms._schemas import ProbeResult


def run_probe(
    *,
    model: nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
) -> ProbeResult:
    threshold = float(config.get("threshold", 0.90))
    n_samples = int(config.get("n_samples", min(32, len(items))))
    shift_pixels = int(config.get("shift_pixels", 8))
    pixel_tolerance = float(config.get("pixel_tolerance", 1.0))
    pixel_size_lj = float(config.get("pixel_size_lj", 0.15))

    if not items:
        return ProbeResult(
            constraint_name="translation_equivariance",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_matched_within_tolerance",
            passes_threshold=False, threshold=threshold,
            details={"reason": "no items provided"},
        )

    model.eval()
    expected_shift_lj = shift_pixels * pixel_size_lj

    matched = 0
    total = 0
    with torch.no_grad():
        for i in range(min(n_samples, len(items))):
            image = items[i]["image"].to(device)
            shifted = torch.roll(image, shifts=(shift_pixels, shift_pixels), dims=(-2, -1))
            batch = torch.stack([image, shifted], dim=0)
            out = model(batch)

            pos_a = out["positions"][0].detach().cpu().numpy()
            pos_b = out["positions"][1].detach().cpu().numpy()
            conf_a = torch.sigmoid(out["confidence_logits"][0]).detach().cpu().numpy()

            top_q = np.argsort(-conf_a)[: max(1, int((conf_a > 0.5).sum()))]
            if top_q.size == 0:
                continue
            sel_a = pos_a[top_q]
            cost = np.linalg.norm(sel_a[:, None, :] - pos_b[None, :, :], axis=-1)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind, strict=True):
                # Equivariance prediction: pos_b ~ pos_a + (-dx, +dy) in LJ
                # because shifting the image right increases column index
                # and our convention maps col to +x and row to -y.
                expected = sel_a[r] + np.array([
                    +shift_pixels * pixel_size_lj,
                    -shift_pixels * pixel_size_lj,
                ])
                err_lj = float(np.linalg.norm(pos_b[c] - expected))
                err_pixels = err_lj / pixel_size_lj
                total += 1
                if err_pixels <= pixel_tolerance:
                    matched += 1

    score = matched / max(1, total)
    return ProbeResult(
        constraint_name="translation_equivariance",
        satisfaction_score=score,
        num_test_cases=total,
        metric="frac_matched_within_tolerance",
        passes_threshold=score >= threshold,
        threshold=threshold,
        details={
            "shift_pixels": shift_pixels,
            "expected_shift_lj": expected_shift_lj,
            "pixel_tolerance": pixel_tolerance,
            "matched": matched,
        },
    )


__all__ = ["run_probe"]
