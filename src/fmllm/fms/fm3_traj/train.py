"""Training loop for FM3 (trajectory -> Gamma KE distribution).

The trainer minimizes the negative log-likelihood of the observed
per-atom-per-frame kinetic energies under the predicted
``Gamma(alpha, beta)``, plus a soft equipartition penalty that pulls
``alpha * beta`` toward the empirical mean kinetic energy.

Produces:
    A checkpoint at ``checkpoints/fm3_traj/<run_id>/model.pt`` plus
    a manifest YAML next to it.

Depends on:
    torch, loguru.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Gamma
from torch.utils.data import DataLoader

from fmllm.fms.common import (
    kinetic_energies_masked,
    make_dataloaders,
    make_optimizer_and_schedule,
    save_checkpoint,
)
from fmllm.fms.fm3_traj.model import FM3TrajTransformer, build_fm3_model
from fmllm.utils.config import Config, FM3Config
from fmllm.utils.logging import configure_logging
from fmllm.utils.manifests import write_manifest
from fmllm.utils.run_ids import generate_run_id


def gamma_nll(
    alpha: Tensor,
    beta: Tensor,
    *,
    samples: Tensor,
    sample_mask: Tensor,
    nll_clip: float,
) -> Tensor:
    """Mean negative log-likelihood of ``samples`` under ``Gamma(alpha, beta)``.

    ``alpha`` and ``beta`` carry shape ``(B,)``. ``samples`` and
    ``sample_mask`` carry shape ``(B, S)`` where ``S`` is the number
    of (atom, time) entries per specimen. Padded entries get masked
    out before averaging.
    """
    safe_samples = samples.clamp_min(1.0e-8)
    dist = Gamma(alpha.unsqueeze(-1), 1.0 / beta.unsqueeze(-1).clamp_min(1.0e-8))
    log_prob = dist.log_prob(safe_samples)
    log_prob = torch.nan_to_num(log_prob, nan=-nll_clip, posinf=-nll_clip, neginf=-nll_clip)
    log_prob = log_prob.clamp(min=-nll_clip, max=nll_clip)
    log_prob = log_prob.masked_fill(~sample_mask, 0.0)

    n_real = sample_mask.sum(dim=-1).clamp(min=1).to(log_prob.dtype)
    per_batch_mean_lp = log_prob.sum(dim=-1) / n_real
    return -per_batch_mean_lp.mean()


def compute_fm3_losses(
    outputs: dict[str, Tensor],
    *,
    samples: Tensor,
    sample_mask: Tensor,
    cfg: FM3Config,
) -> dict[str, Tensor]:
    """NLL plus equipartition penalty for FM3."""
    alpha = outputs["alpha"]
    beta = outputs["beta"]

    nll = gamma_nll(
        alpha, beta,
        samples=samples,
        sample_mask=sample_mask,
        nll_clip=cfg.nll_clip,
    )

    n_real = sample_mask.sum(dim=-1).clamp(min=1).to(samples.dtype)
    safe_samples = samples.masked_fill(~sample_mask, 0.0)
    empirical_mean = safe_samples.sum(dim=-1) / n_real
    pred_mean = alpha * beta
    equi = ((pred_mean - empirical_mean) ** 2).mean()

    total = nll + cfg.equipartition_weight * equi
    return {
        "total": total,
        "nll": nll.detach(),
        "equipartition": equi.detach(),
        "pred_mean_ke": pred_mean.mean().detach(),
        "obs_mean_ke": empirical_mean.mean().detach(),
    }


def _flatten_kinetic_energies(
    velocities: Tensor,
    atom_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute KE per (atom, frame), then flatten to ``(B, S)`` plus mask."""
    ke, mask = kinetic_energies_masked(velocities, atom_mask)
    b, t, n = ke.shape
    return ke.reshape(b, t * n), mask.reshape(b, t * n)


def _prepare_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    cfg: FM3Config,
) -> dict[str, Tensor]:
    traj_pos = batch["traj_positions"].to(device, non_blocking=True)
    traj_vel = batch["traj_velocities"].to(device, non_blocking=True)
    atom_mask = batch["atom_mask"].to(device, non_blocking=True)

    traj_pos = traj_pos[:, :, : cfg.max_n_atoms]
    traj_vel = traj_vel[:, :, : cfg.max_n_atoms]
    atom_mask = atom_mask[:, : cfg.max_n_atoms]

    ke_flat, ke_mask = _flatten_kinetic_energies(traj_vel, atom_mask)
    return {
        "traj_positions": traj_pos,
        "traj_velocities": traj_vel,
        "atom_mask": atom_mask,
        "samples": ke_flat,
        "sample_mask": ke_mask,
    }


def _epoch(
    model: FM3TrajTransformer,
    loader: DataLoader,
    *,
    cfg: FM3Config,
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

    sums = {"total": 0.0, "nll": 0.0, "equipartition": 0.0,
            "pred_mean_ke": 0.0, "obs_mean_ke": 0.0}
    count = 0
    for batch in loader:
        prepared = _prepare_batch(batch, device=device, cfg=cfg)
        with torch.set_grad_enabled(train):
            if use_amp and train:
                with autocast(dtype=torch.float16):
                    outputs = model(
                        prepared["traj_positions"],
                        prepared["traj_velocities"],
                        prepared["atom_mask"],
                    )
                    losses = compute_fm3_losses(
                        outputs,
                        samples=prepared["samples"],
                        sample_mask=prepared["sample_mask"],
                        cfg=cfg,
                    )
            else:
                outputs = model(
                    prepared["traj_positions"],
                    prepared["traj_velocities"],
                    prepared["atom_mask"],
                )
                losses = compute_fm3_losses(
                    outputs,
                    samples=prepared["samples"],
                    sample_mask=prepared["sample_mask"],
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

        bs = prepared["samples"].shape[0]
        for key in sums:
            sums[key] += float(losses[key].detach()) * bs
        count += bs

    return {k: v / max(1, count) for k, v in sums.items()}


def train(
    *,
    cfg: Config,
    h5_path: Path | str,
    splits_path: Path | str,
    out_dir: Path | str | None = None,
    device: str | torch.device = "auto",
    epochs: int | None = None,
) -> Path:
    fm = cfg.fm3
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
    logger.info("FM3 training: device={}, epochs={}, out={}", dev, epochs, out_dir)

    train_loader, val_loader, calib_loader, sub = make_dataloaders(
        h5_path, splits_path,
        batch_size=fm.batch_size,
        num_workers=fm.num_workers,
        val_fraction=fm.val_fraction,
        calib_fraction=fm.calib_fraction,
        keys=("traj_positions", "traj_velocities", "atom_mask", "specimen_id"),
        seed=cfg.seeds.numpy,
    )
    logger.info(
        "Dataset split: train={}, val={}, calib={}",
        len(train_loader.dataset), len(val_loader.dataset), len(calib_loader.dataset),
    )

    model = build_fm3_model(fm).to(dev)
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
            "  train: total={:.4f} nll={:.4f} equi={:.4f} pred_KE={:.3f} obs_KE={:.3f}",
            train_metrics["total"], train_metrics["nll"],
            train_metrics["equipartition"],
            train_metrics["pred_mean_ke"], train_metrics["obs_mean_ke"],
        )
        logger.info(
            "  val:   total={:.4f} nll={:.4f} equi={:.4f} pred_KE={:.3f} obs_KE={:.3f}",
            val_metrics["total"], val_metrics["nll"],
            val_metrics["equipartition"],
            val_metrics["pred_mean_ke"], val_metrics["obs_mean_ke"],
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
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
        script="fmllm.fms.fm3_traj.train",
        inputs={"h5_path": str(h5_path), "splits_path": str(splits_path)},
        config={
            "fm3": fm.model_dump(),
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


__all__ = ["compute_fm3_losses", "gamma_nll", "train"]
