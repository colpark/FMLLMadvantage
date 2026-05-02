"""Split-conformal calibration for FM2 energy predictions.

The non-conformity score per specimen is the absolute residual
``|E_pred - E_true|``. Calibration fits a per-alpha threshold on the
calibration subset of the training pool. At inference, the prediction
interval at miscoverage level ``alpha`` is ``E_pred +/- q_alpha``.

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

from fmllm.fms.common import (
    load_checkpoint,
    make_dataloaders,
    split_conformal_quantile,
    write_conformal_calibration,
)
from fmllm.fms.fm2_rdf.model import build_fm2_model
from fmllm.fms.fm2_rdf.train import _prepare_batch
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
    fm = cfg.fm2
    confinement_k = cfg.dataset.confinement_k
    max_n_atoms = max(cfg.dataset.n_choices)

    if isinstance(device, str):
        dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device)
        if device == "auto" and not torch.cuda.is_available():
            dev = torch.device("cpu")
    else:
        dev = device

    model = build_fm2_model(fm).to(dev)
    load_checkpoint(checkpoint_path, model=model, map_location=dev)
    model.eval()

    _, _, calib_loader, _ = make_dataloaders(
        h5_path, splits_path,
        batch_size=fm.batch_size,
        num_workers=fm.num_workers,
        val_fraction=fm.val_fraction,
        calib_fraction=fm.calib_fraction,
        keys=("rdf", "traj_positions", "atom_count", "atom_mask", "specimen_id"),
        seed=cfg.seeds.numpy,
    )

    residuals: list[float] = []
    with torch.no_grad():
        for batch in calib_loader:
            prepared = _prepare_batch(
                batch,
                cfg_dataset_confinement_k=confinement_k,
                cfg_max_n_atoms=max_n_atoms,
                device=dev,
            )
            pred = model(prepared["rdf"])
            res = (pred - prepared["target_energy"]).abs().detach().cpu().numpy()
            residuals.extend(float(x) for x in res.tolist())

    if not residuals:
        raise RuntimeError("calibration set produced no residuals")
    scores = np.asarray(residuals, dtype=np.float32)

    alpha_to_q = {
        float(a): split_conformal_quantile(scores, a)
        for a in fm.conformal_alpha_levels
    }
    logger.info("FM2 conformal thresholds: {}", alpha_to_q)

    if out_path is None:
        out_path = Path(checkpoint_path).parent / "calibration.json"
    return write_conformal_calibration(
        out_path,
        fm_name=fm.name,
        score_name="energy_abs_residual",
        alpha_to_threshold=alpha_to_q,
        extra={
            "num_calibration_specimens": int(scores.size),
            "score_summary": {
                "mean": float(scores.mean()),
                "median": float(np.median(scores)),
                "p90": float(np.quantile(scores, 0.90)),
                "p95": float(np.quantile(scores, 0.95)),
                "max": float(scores.max()),
            },
        },
    )


__all__ = ["calibrate"]
