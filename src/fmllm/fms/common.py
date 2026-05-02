"""Shared utilities for FM training, conformal calibration, and I/O.

This module hosts helpers each FM trainer reuses: data-loading from the
HDF5 store with deterministic train / val / calibration splits drawn
from the training partition, the AdamW + cosine-with-warmup schedule
the project standardizes on, AMP scaffolding, checkpoint I/O, and
split-conformal quantile calibration.

Produces:
    Reusable building blocks. The FM-specific train scripts wire them
    into per-FM forward passes and loss functions.

Depends on:
    torch, numpy, loguru, pyyaml.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from fmllm.data.dataset import LJSpecimenDataset
from fmllm.data.splits import load_splits_yaml, select_train_subset


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def deterministic_train_val_calib_split(
    train_ids: Sequence[int],
    *,
    val_fraction: float,
    calib_fraction: float,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Carve the training population into train / val / calibration subsets.

    The carve uses a fixed seed so every FM trainer sees the same
    partition of the dataset's training pool. The function returns
    sorted lists for stable iteration order.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}")
    if not (0.0 < calib_fraction < 1.0):
        raise ValueError(f"calib_fraction must lie in (0, 1), got {calib_fraction}")
    if val_fraction + calib_fraction >= 1.0:
        raise ValueError(
            "val_fraction + calib_fraction must stay below 1.0"
        )

    rng = np.random.default_rng(seed)
    ids = np.asarray(list(train_ids), dtype=np.int64)
    perm = rng.permutation(ids.shape[0])
    n_val = int(round(val_fraction * ids.shape[0]))
    n_calib = int(round(calib_fraction * ids.shape[0]))
    val_idx = perm[:n_val]
    calib_idx = perm[n_val : n_val + n_calib]
    train_idx = perm[n_val + n_calib :]
    return {
        "train": sorted(int(x) for x in ids[train_idx]),
        "val": sorted(int(x) for x in ids[val_idx]),
        "calib": sorted(int(x) for x in ids[calib_idx]),
    }


def make_dataloaders(
    h5_path: Path | str,
    splits_path: Path | str,
    *,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    calib_fraction: float,
    keys: Iterable[str] | None = None,
    seed: int = 0,
    train_split: str = "train_full",
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[int]]]:
    """Open the dataset and return train, val, calibration loaders.

    The split assignment uses :func:`deterministic_train_val_calib_split`
    so every FM gets the same partition of the training pool. The
    holdout split that lives in ``splits_path`` stays untouched here;
    Phase 7 reads it for the world-model evaluation tests.

    Args:
        train_split: Selects which nested training subset to use.
            ``"train_full"`` (the default) uses the entire training
            pool. ``"train_10k"``, ``"train_30k"``, ``"train_50k"``
            select the nested subsets the splits YAML records under
            ``train_subsets``. The FM-quality sweep (E5) trains each
            FM at all three scales.
    """
    splits = load_splits_yaml(splits_path)
    train_pool = select_train_subset(splits, train_split)
    if not train_pool:
        raise ValueError(
            f"splits file {splits_path} produced empty train pool for "
            f"train_split={train_split!r}"
        )

    sub = deterministic_train_val_calib_split(
        train_pool,
        val_fraction=val_fraction,
        calib_fraction=calib_fraction,
        seed=seed,
    )

    train_ds = LJSpecimenDataset(h5_path, specimen_ids=sub["train"], keys=keys)
    val_ds = LJSpecimenDataset(h5_path, specimen_ids=sub["val"], keys=keys)
    calib_ds = LJSpecimenDataset(h5_path, specimen_ids=sub["calib"], keys=keys)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    calib_loader = DataLoader(calib_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, calib_loader, sub


# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------


def make_optimizer_and_schedule(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> tuple[torch.optim.Optimizer, LambdaLR]:
    """AdamW + linear warmup + cosine decay to zero.

    The schedule operates per optimizer step. ``warmup_epochs`` worth
    of steps ramp linearly from zero to ``learning_rate``. The remainder
    decays as a half cosine to zero at the final step.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save model weights and training state to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` (and optionally optim/sched)."""
    path = Path(path)
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


def split_conformal_quantile(scores: torch.Tensor | np.ndarray, alpha: float) -> float:
    """Return the split-conformal threshold at miscoverage level ``alpha``.

    For ``n`` calibration scores, the threshold is the
    ``ceil((n + 1) * (1 - alpha))``-th smallest score (1-indexed) with
    the standard finite-sample correction.
    """
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)
    if scores_np.ndim != 1:
        scores_np = scores_np.reshape(-1)
    if scores_np.size == 0:
        raise ValueError("split_conformal_quantile got empty scores")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    sorted_scores = np.sort(scores_np)
    n = sorted_scores.shape[0]
    k = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    k = max(0, min(n - 1, k))
    return float(sorted_scores[k])


def write_conformal_calibration(
    path: Path | str,
    *,
    fm_name: str,
    score_name: str,
    alpha_to_threshold: dict[float, float],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a conformal-calibration JSON file with the standard schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "fm_name": fm_name,
        "score_name": score_name,
        "thresholds": {f"{a:.4f}": v for a, v in alpha_to_threshold.items()},
        "extra": extra or {},
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def read_conformal_calibration(path: Path | str) -> dict[str, Any]:
    """Read a conformal-calibration JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"calibration file not found: {path}")
    with path.open("r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Targets and helpers
# ---------------------------------------------------------------------------


def per_atom_potential_energy(
    final_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    confinement_k: float,
) -> torch.Tensor:
    """Return per-atom potential energy at the supplied positions.

    The function evaluates the 12-6 LJ pair sum with pair-level masking
    so padded atoms drop out cleanly, then adds the harmonic-confinement
    contribution from real atoms only. Returns one scalar per leading
    batch entry.
    """
    n_max = final_positions.shape[1]
    diff = final_positions.unsqueeze(-2) - final_positions.unsqueeze(-3)
    r2 = (diff * diff).sum(dim=-1)

    eye = torch.eye(n_max, dtype=torch.bool, device=final_positions.device)
    pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
    pair_mask = pair_mask & ~eye

    r2_safe = torch.where(pair_mask, r2, torch.ones_like(r2)).clamp_min(1.0e-12)
    inv_r2 = 1.0 / r2_safe
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    pair_energy = torch.where(pair_mask, pair_energy, torch.zeros_like(pair_energy))
    lj_energy = 0.5 * pair_energy.sum(dim=(-1, -2))

    pos_real = final_positions * atom_mask.unsqueeze(-1).to(final_positions.dtype)
    conf_energy = 0.5 * confinement_k * (pos_real * pos_real).sum(dim=(-1, -2))

    n_real = atom_mask.sum(dim=-1).clamp(min=1).to(final_positions.dtype)
    return (lj_energy + conf_energy) / n_real


def kinetic_energies_masked(
    velocities: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-atom-per-frame kinetic energies with padding masked out.

    Returns ``ke`` of shape ``(B, T, N)`` and a parallel ``mask`` of
    the same shape so callers can stack KEs across the batch when
    computing per-specimen empirical statistics.
    """
    ke = 0.5 * (velocities * velocities).sum(dim=-1)  # (B, T, N)
    mask = atom_mask.unsqueeze(1).expand(-1, ke.shape[1], -1)
    return ke, mask


def gather_specimen_metadata(
    h5_path: Path | str,
    specimen_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    """Read scalar per-specimen metadata for a list of IDs."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        ids = np.asarray(list(specimen_ids), dtype=np.int64)
        return {
            "atom_counts": np.asarray(f["atom_counts"])[ids].astype(np.int64),
            "temperatures": np.asarray(f["temperatures"])[ids].astype(np.float32),
            "motif_ids": np.asarray(f["motif_ids"])[ids].astype(np.int64),
        }


__all__ = [
    "deterministic_train_val_calib_split",
    "gather_specimen_metadata",
    "kinetic_energies_masked",
    "load_checkpoint",
    "make_dataloaders",
    "make_optimizer_and_schedule",
    "per_atom_potential_energy",
    "read_conformal_calibration",
    "save_checkpoint",
    "split_conformal_quantile",
    "write_conformal_calibration",
]


