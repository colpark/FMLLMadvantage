"""Training loop for FM1 (image -> count + atom positions).

The trainer minimizes a sum of:

    - count cross-entropy on the categorical atom-count head,
    - Hungarian-matched L2 position loss against the ground-truth
      atom positions in the final trajectory frame,
    - binary cross-entropy on the confidence (objectness) logits with
      Hungarian-matched targets,
    - a soft box-constraint loss that penalizes predicted positions
      outside the imaging box.

The trainer logs running loss components per epoch, validates every
epoch on a held-out validation subset of the training pool, and saves
the checkpoint with the best validation total loss.

Produces:
    A checkpoint at ``checkpoints/fm1_image/<run_id>/model.pt`` plus
    a manifest YAML next to it.

Depends on:
    torch, scipy, numpy, loguru.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from fmllm.fms.common import (
    make_dataloaders,
    make_optimizer_and_schedule,
    save_checkpoint,
)
from fmllm.fms.fm1_image.model import FM1ImageViT, build_fm1_model
from fmllm.utils.config import Config, FM1Config
from fmllm.utils.logging import configure_logging
from fmllm.utils.manifests import write_manifest
from fmllm.utils.run_ids import generate_run_id


def hungarian_match_batch(
    pred_positions: Tensor,
    true_positions: Tensor,
    atom_mask: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    """For each batch entry, find the lowest-cost matching of queries to atoms.

    Args:
        pred_positions: ``(B, Q, 2)`` predicted positions.
        true_positions: ``(B, N_max, 2)`` ground-truth positions.
        atom_mask: ``(B, N_max)`` boolean mask flagging real atoms.

    Returns:
        Two parallel lists of length ``B``, each entry a 1D long tensor.
        ``pred_idx[b]`` and ``true_idx[b]`` index into the query slots
        and the real-atom slots respectively.
    """
    pred_np = pred_positions.detach().cpu().numpy()
    true_np = true_positions.detach().cpu().numpy()
    mask_np = atom_mask.detach().cpu().numpy()

    pred_indices: list[Tensor] = []
    true_indices: list[Tensor] = []
    for b in range(pred_np.shape[0]):
        n_real = int(mask_np[b].sum())
        if n_real == 0:
            pred_indices.append(torch.empty(0, dtype=torch.long))
            true_indices.append(torch.empty(0, dtype=torch.long))
            continue
        true_b = true_np[b, :n_real]
        cost = np.linalg.norm(
            pred_np[b][:, None, :] - true_b[None, :, :],
            axis=-1,
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        pred_indices.append(torch.from_numpy(np.asarray(row_ind, dtype=np.int64)))
        true_indices.append(torch.from_numpy(np.asarray(col_ind, dtype=np.int64)))
    return pred_indices, true_indices


def compute_fm1_losses(
    outputs: dict[str, Tensor],
    *,
    target_count: Tensor,
    target_positions: Tensor,
    atom_mask: Tensor,
    cfg: FM1Config,
) -> dict[str, Tensor]:
    """Compute the FM1 training loss decomposition.

    Returns a dict with ``total``, ``count``, ``position``,
    ``confidence``, ``box`` losses, plus the matched-pair count.
    """
    count_logits = outputs["count_logits"]
    pred_positions = outputs["positions"]
    pred_conf_logits = outputs["confidence_logits"]
    device = pred_positions.device
    batch = pred_positions.shape[0]
    num_queries = pred_positions.shape[1]

    count_loss = nn.functional.cross_entropy(count_logits, target_count)

    pred_idx, true_idx = hungarian_match_batch(
        pred_positions, target_positions, atom_mask,
    )

    matched_pred: list[Tensor] = []
    matched_true: list[Tensor] = []
    confidence_targets = torch.zeros(
        batch, num_queries, device=device, dtype=pred_conf_logits.dtype,
    )
    total_matched = 0
    for b, (pi, ti) in enumerate(zip(pred_idx, true_idx, strict=True)):
        if pi.numel() == 0:
            continue
        matched_pred.append(pred_positions[b, pi.to(device)])
        matched_true.append(target_positions[b, ti.to(device)])
        confidence_targets[b, pi.to(device)] = 1.0
        total_matched += int(pi.numel())

    if matched_pred:
        position_loss = nn.functional.mse_loss(
            torch.cat(matched_pred, dim=0),
            torch.cat(matched_true, dim=0),
        )
    else:
        position_loss = pred_positions.new_zeros(())

    confidence_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_conf_logits, confidence_targets,
    )

    half_w = cfg.box_half_width_lj
    box_excess = torch.relu(pred_positions.abs() - half_w)
    box_loss = (box_excess * box_excess).mean()

    total = (
        cfg.count_weight * count_loss
        + cfg.position_weight * position_loss
        + cfg.confidence_weight * confidence_loss
        + cfg.box_constraint_weight * box_loss
    )

    return {
        "total": total,
        "count": count_loss.detach(),
        "position": position_loss.detach(),
        "confidence": confidence_loss.detach(),
        "box": box_loss.detach(),
        "matched_pairs": torch.tensor(total_matched, device=device),
    }


def _prepare_batch(
    batch: dict[str, Any],
    *,
    cfg: FM1Config,
    device: torch.device,
) -> dict[str, Tensor]:
    image = batch["image"].to(device, non_blocking=True)
    traj_pos = batch["traj_positions"].to(device, non_blocking=True)
    atom_mask = batch["atom_mask"].to(device, non_blocking=True)
    target_count = batch["atom_count"].to(device, non_blocking=True).long()
    final_positions = traj_pos[:, -1]
    final_positions = final_positions[:, : cfg.max_n_atoms]
    atom_mask = atom_mask[:, : cfg.max_n_atoms]
    return {
        "image": image,
        "target_positions": final_positions,
        "atom_mask": atom_mask,
        "target_count": target_count.clamp_max(cfg.max_n_atoms),
    }


def _epoch(
    model: FM1ImageViT,
    loader: DataLoader,
    *,
    cfg: FM1Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: GradScaler | None,
    train: bool,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    use_amp = cfg.mixed_precision and device.type == "cuda"

    sums = {"total": 0.0, "count": 0.0, "position": 0.0, "confidence": 0.0, "box": 0.0}
    count = 0
    matched_total = 0
    for batch in loader:
        prepared = _prepare_batch(batch, cfg=cfg, device=device)
        with torch.set_grad_enabled(train):
            if use_amp and train:
                with autocast(dtype=torch.float16):
                    outputs = model(prepared["image"])
                    losses = compute_fm1_losses(
                        outputs,
                        target_count=prepared["target_count"],
                        target_positions=prepared["target_positions"],
                        atom_mask=prepared["atom_mask"],
                        cfg=cfg,
                    )
            else:
                outputs = model(prepared["image"])
                losses = compute_fm1_losses(
                    outputs,
                    target_count=prepared["target_count"],
                    target_positions=prepared["target_positions"],
                    atom_mask=prepared["atom_mask"],
                    cfg=cfg,
                )

        if train:
            assert optimizer is not None and scheduler is not None
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            scheduler.step()

        bs = prepared["image"].shape[0]
        for key in sums:
            sums[key] += float(losses[key].detach()) * bs
        matched_total += int(losses["matched_pairs"].item())
        count += bs

    return {
        **{k: v / max(1, count) for k, v in sums.items()},
        "matched_per_sample": matched_total / max(1, count),
    }


def train(
    *,
    cfg: Config,
    h5_path: Path | str,
    splits_path: Path | str,
    out_dir: Path | str | None = None,
    device: str | torch.device = "auto",
    epochs: int | None = None,
) -> Path:
    """Run FM1 training end to end and return the checkpoint path."""
    fm = cfg.fm1
    epochs = epochs if epochs is not None else fm.epochs

    if isinstance(device, str):
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
    else:
        dev = device

    run_id = generate_run_id(f"{fm.name}-train")
    run_dir = Path("runs") / run_id
    configure_logging(run_dir)
    if out_dir is None:
        out_dir = Path(fm.checkpoint_root) / fm.name / run_id
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("FM1 training: device={}, epochs={}, out={}", dev, epochs, out_dir)

    train_loader, val_loader, calib_loader, sub = make_dataloaders(
        h5_path, splits_path,
        batch_size=fm.batch_size,
        num_workers=fm.num_workers,
        val_fraction=fm.val_fraction,
        calib_fraction=fm.calib_fraction,
        keys=("image", "traj_positions", "atom_count", "atom_mask", "specimen_id"),
        seed=cfg.seeds.numpy,
    )
    logger.info(
        "Dataset split: train={}, val={}, calib={}",
        len(train_loader.dataset), len(val_loader.dataset), len(calib_loader.dataset),
    )

    model = build_fm1_model(fm).to(dev)
    optimizer, scheduler = make_optimizer_and_schedule(
        model,
        learning_rate=fm.learning_rate,
        weight_decay=fm.weight_decay,
        epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)),
        warmup_epochs=fm.warmup_epochs,
    )
    scaler = GradScaler() if (fm.mixed_precision and dev.type == "cuda") else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameter count: {:.2f}M", n_params / 1.0e6)

    best_val = math.inf
    best_path = out_dir / "model.pt"
    history: list[dict[str, float]] = []

    t_start = time.time()
    for epoch in range(epochs):
        logger.info("Epoch {}/{}", epoch + 1, epochs)
        train_metrics = _epoch(
            model, train_loader, cfg=fm, device=dev,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, train=True,
        )
        val_metrics = _epoch(
            model, val_loader, cfg=fm, device=dev,
            optimizer=None, scheduler=None, scaler=None, train=False,
        )
        logger.info(
            "  train: total={:.4f} count={:.4f} pos={:.4f} conf={:.4f} box={:.4f}",
            train_metrics["total"], train_metrics["count"],
            train_metrics["position"], train_metrics["confidence"],
            train_metrics["box"],
        )
        logger.info(
            "  val:   total={:.4f} count={:.4f} pos={:.4f} conf={:.4f} box={:.4f}",
            val_metrics["total"], val_metrics["count"],
            val_metrics["position"], val_metrics["confidence"],
            val_metrics["box"],
        )
        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(
                best_path,
                model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1,
                extra={"val_metrics": val_metrics, "fm_config": fm.model_dump()},
            )
            logger.info("  best so far. saved to {}", best_path)

    elapsed = time.time() - t_start
    logger.info("Training done in {:.1f}s. Best val total: {:.4f}", elapsed, best_val)

    write_manifest(
        out_dir / "manifest.yaml",
        script="fmllm.fms.fm1_image.train",
        inputs={"h5_path": str(h5_path), "splits_path": str(splits_path)},
        config={
            "fm1": fm.model_dump(),
            "run_id": run_id,
            "device": str(dev),
            "epochs": epochs,
        },
        extra={
            "n_params": n_params,
            "best_val_total": best_val,
            "split_sizes": {k: len(v) for k, v in sub.items()},
            "elapsed_seconds": elapsed,
            "history": history,
        },
    )
    return best_path


__all__ = ["compute_fm1_losses", "hungarian_match_batch", "train"]
