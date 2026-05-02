"""Split-conformal calibration for FM1 position predictions.

The non-conformity score per matched (predicted, true) pair is the L2
distance between the predicted position and the matched ground-truth
position. Calibration fits a per-alpha threshold on the calibration
subset of the training pool. At inference, an 80% (resp. 90%)
prediction circle around each predicted atom position has radius
``q_{alpha=0.20}`` (resp. ``q_{alpha=0.10}``).

Produces:
    A JSON file at ``calibration.json`` next to the model checkpoint.

Depends on:
    torch, numpy, loguru.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from fmllm.fms.common import (
    load_checkpoint,
    make_dataloaders,
    split_conformal_quantile,
    write_conformal_calibration,
)
from fmllm.fms.fm1_image.model import build_fm1_model
from fmllm.fms.fm1_image.train import (
    _prepare_batch,
    hungarian_match_batch,
)
from fmllm.utils.config import Config


def calibrate(
    *,
    cfg: Config,
    checkpoint_path: Path | str,
    h5_path: Path | str,
    splits_path: Path | str,
    out_path: Path | str | None = None,
    device: str | torch.device = "auto",
) -> Path:
    """Fit split-conformal thresholds for FM1 position predictions."""
    fm = cfg.fm1

    if isinstance(device, str):
        dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device)
        if device == "auto" and not torch.cuda.is_available():
            dev = torch.device("cpu")
    else:
        dev = device

    model = build_fm1_model(fm).to(dev)
    load_checkpoint(checkpoint_path, model=model, map_location=dev)
    model.eval()

    _, _, calib_loader, _ = make_dataloaders(
        h5_path, splits_path,
        batch_size=fm.batch_size,
        num_workers=fm.num_workers,
        val_fraction=fm.val_fraction,
        calib_fraction=fm.calib_fraction,
        keys=("image", "traj_positions", "atom_count", "atom_mask", "specimen_id"),
        seed=cfg.seeds.numpy,
    )

    scores: list[float] = []
    with torch.no_grad():
        for batch in calib_loader:
            prepared = _prepare_batch(batch, cfg=fm, device=dev)
            outputs = model(prepared["image"])
            pred_positions = outputs["positions"]
            true_positions = prepared["target_positions"]
            atom_mask = prepared["atom_mask"]
            pred_idx, true_idx = hungarian_match_batch(
                pred_positions, true_positions, atom_mask,
            )
            for b, (pi, ti) in enumerate(zip(pred_idx, true_idx, strict=True)):
                if pi.numel() == 0:
                    continue
                pp = pred_positions[b, pi.to(dev)].detach().cpu()
                tp = true_positions[b, ti.to(dev)].detach().cpu()
                d = (pp - tp).norm(dim=-1)
                scores.extend(float(x) for x in d.tolist())

    if not scores:
        raise RuntimeError("calibration set produced no matched pairs")
    scores_np = np.asarray(scores, dtype=np.float32)

    alpha_to_q = {
        float(a): split_conformal_quantile(scores_np, a)
        for a in fm.conformal_alpha_levels
    }
    logger.info("FM1 conformal thresholds: {}", alpha_to_q)

    if out_path is None:
        out_path = Path(checkpoint_path).parent / "calibration.json"
    return write_conformal_calibration(
        out_path,
        fm_name=fm.name,
        score_name="position_l2_lj",
        alpha_to_threshold=alpha_to_q,
        extra={
            "num_calibration_pairs": int(scores_np.size),
            "score_summary": {
                "mean": float(scores_np.mean()),
                "median": float(np.median(scores_np)),
                "p90": float(np.quantile(scores_np, 0.90)),
                "p95": float(np.quantile(scores_np, 0.95)),
                "max": float(scores_np.max()),
            },
        },
    )


__all__ = ["calibrate"]
