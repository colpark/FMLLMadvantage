"""Split-conformal calibration for FM3 Gamma KE distribution predictions.

The non-conformity score per specimen is the negative log-likelihood
of the empirical KE distribution under the predicted ``Gamma(alpha,
beta)``, averaged across (atom, frame) pairs. Calibration fits a
threshold per alpha. At inference, a specimen with score below
``q_alpha`` falls inside the calibrated band at miscoverage level
``alpha``.

Produces:
    A JSON file at ``calibration.json`` next to the model checkpoint.

Depends on:
    torch, numpy, loguru.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.distributions import Gamma

from fmllm.fms.common import (
    load_checkpoint,
    make_dataloaders,
    split_conformal_quantile,
    write_conformal_calibration,
)
from fmllm.fms.fm3_traj.model import build_fm3_model
from fmllm.fms.fm3_traj.train import _prepare_batch
from fmllm.utils.config import Config


def _per_specimen_nll(alpha, beta, samples, sample_mask, nll_clip):
    safe_samples = samples.clamp_min(1.0e-8)
    dist = Gamma(alpha.unsqueeze(-1), 1.0 / beta.unsqueeze(-1).clamp_min(1.0e-8))
    log_prob = dist.log_prob(safe_samples)
    log_prob = torch.nan_to_num(log_prob, nan=-nll_clip, posinf=-nll_clip, neginf=-nll_clip)
    log_prob = log_prob.clamp(min=-nll_clip, max=nll_clip)
    log_prob = log_prob.masked_fill(~sample_mask, 0.0)
    n = sample_mask.sum(dim=-1).clamp(min=1).to(log_prob.dtype)
    return -log_prob.sum(dim=-1) / n


def calibrate(
    *,
    cfg: Config,
    checkpoint_path: Path | str,
    h5_path: Path | str,
    splits_path: Path | str,
    out_path: Path | str | None = None,
    device: str | torch.device = "auto",
) -> Path:
    fm = cfg.fm3

    if isinstance(device, str):
        dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device)
        if device == "auto" and not torch.cuda.is_available():
            dev = torch.device("cpu")
    else:
        dev = device

    model = build_fm3_model(fm).to(dev)
    load_checkpoint(checkpoint_path, model=model, map_location=dev)
    model.eval()

    _, _, calib_loader, _ = make_dataloaders(
        h5_path, splits_path,
        batch_size=fm.batch_size,
        num_workers=fm.num_workers,
        val_fraction=fm.val_fraction,
        calib_fraction=fm.calib_fraction,
        keys=("traj_positions", "traj_velocities", "atom_mask", "specimen_id"),
        seed=cfg.seeds.numpy,
    )

    scores: list[float] = []
    with torch.no_grad():
        for batch in calib_loader:
            prepared = _prepare_batch(batch, device=dev, cfg=fm)
            outputs = model(
                prepared["traj_positions"],
                prepared["traj_velocities"],
                prepared["atom_mask"],
            )
            nll = _per_specimen_nll(
                outputs["alpha"], outputs["beta"],
                prepared["samples"], prepared["sample_mask"],
                fm.nll_clip,
            )
            scores.extend(float(x) for x in nll.detach().cpu().tolist())

    if not scores:
        raise RuntimeError("calibration set produced no scores")
    scores_np = np.asarray(scores, dtype=np.float32)

    alpha_to_q = {
        float(a): split_conformal_quantile(scores_np, a)
        for a in fm.conformal_alpha_levels
    }
    logger.info("FM3 conformal thresholds: {}", alpha_to_q)

    if out_path is None:
        out_path = Path(checkpoint_path).parent / "calibration.json"
    return write_conformal_calibration(
        out_path,
        fm_name=fm.name,
        score_name="ke_distribution_nll",
        alpha_to_threshold=alpha_to_q,
        extra={
            "num_calibration_specimens": int(scores_np.size),
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
