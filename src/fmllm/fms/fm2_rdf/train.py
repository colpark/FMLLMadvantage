"""Training loop for FM2 (RDF -> per-atom potential energy).

The trainer minimizes Huber loss between predicted and ground-truth
per-atom potential energy, plus a soft non-negativity penalty that
discourages predictions below the LJ pair-energy floor (about ``-3``
per atom for our cluster sizes). Extensive scaling holds by output
design, since the model predicts per-atom energy.

Produces:
    A checkpoint at ``checkpoints/fm2_rdf/<run_id>/model.pt`` plus
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
from torch.utils.data import DataLoader

from fmllm.fms.common import (
    make_dataloaders,
    make_optimizer_and_schedule,
    per_atom_potential_energy,
    save_checkpoint,
)
from fmllm.fms.fm2_rdf.model import FM2RDFTransformer, build_fm2_model
from fmllm.utils.config import Config, FM2Config
from fmllm.utils.logging import configure_logging
from fmllm.utils.manifests import write_manifest
from fmllm.utils.run_ids import generate_run_id


def compute_fm2_losses(
    pred_energy: Tensor,
    *,
    target_energy: Tensor,
    cfg: FM2Config,
) -> dict[str, Tensor]:
    """Huber + non-negativity floor loss for FM2."""
    huber = nn.functional.huber_loss(pred_energy, target_energy, delta=cfg.huber_delta)
    floor_violation = torch.relu(cfg.energy_floor - pred_energy)
    nonneg = (floor_violation * floor_violation).mean()
    total = huber + cfg.nonneg_weight * nonneg
    return {
        "total": total,
        "huber": huber.detach(),
        "nonneg": nonneg.detach(),
        "mae": (pred_energy - target_energy).abs().mean().detach(),
    }


def _prepare_batch(
    batch: dict[str, Any],
    *,
    cfg_dataset_confinement_k: float,
    cfg_max_n_atoms: int,
    device: torch.device,
) -> dict[str, Tensor]:
    rdf = batch["rdf"].to(device, non_blocking=True)
    traj_pos = batch["traj_positions"].to(device, non_blocking=True)
    atom_mask = batch["atom_mask"].to(device, non_blocking=True)
    final_positions = traj_pos[:, -1][:, :cfg_max_n_atoms]
    atom_mask = atom_mask[:, :cfg_max_n_atoms]
    target_energy = per_atom_potential_energy(
        final_positions, atom_mask, confinement_k=cfg_dataset_confinement_k,
    )
    return {
        "rdf": rdf,
        "target_energy": target_energy.to(rdf.dtype),
    }


def _epoch(
    model: FM2RDFTransformer,
    loader: DataLoader,
    *,
    cfg: FM2Config,
    confinement_k: float,
    max_n_atoms: int,
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

    sums = {"total": 0.0, "huber": 0.0, "nonneg": 0.0, "mae": 0.0}
    count = 0
    for batch in loader:
        prepared = _prepare_batch(
            batch,
            cfg_dataset_confinement_k=confinement_k,
            cfg_max_n_atoms=max_n_atoms,
            device=device,
        )
        with torch.set_grad_enabled(train):
            if use_amp and train:
                with autocast(dtype=torch.float16):
                    pred = model(prepared["rdf"])
                    losses = compute_fm2_losses(
                        pred, target_energy=prepared["target_energy"], cfg=cfg,
                    )
            else:
                pred = model(prepared["rdf"])
                losses = compute_fm2_losses(
                    pred, target_energy=prepared["target_energy"], cfg=cfg,
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

        bs = prepared["rdf"].shape[0]
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
    train_split: str = "train_full",
) -> Path:
    fm = cfg.fm2
    epochs = epochs if epochs is not None else fm.epochs
    confinement_k = cfg.dataset.confinement_k
    max_n_atoms = max(cfg.dataset.n_choices)

    if isinstance(device, str):
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
    else:
        dev = device

    run_id = generate_run_id(f"{fm.name}-{train_split}-train")
    run_dir = Path("runs") / run_id
    configure_logging(run_dir)
    if out_dir is None:
        out_dir = Path(fm.checkpoint_root) / fm.name / train_split / run_id
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "FM2 training: device={}, epochs={}, train_split={}, out={}",
        dev, epochs, train_split, out_dir,
    )

    train_loader, val_loader, calib_loader, sub = make_dataloaders(
        h5_path, splits_path,
        batch_size=fm.batch_size,
        num_workers=fm.num_workers,
        val_fraction=fm.val_fraction,
        calib_fraction=fm.calib_fraction,
        keys=("rdf", "traj_positions", "atom_count", "atom_mask", "specimen_id"),
        seed=cfg.seeds.numpy,
        train_split=train_split,
    )
    logger.info(
        "Dataset split: train={}, val={}, calib={}",
        len(train_loader.dataset), len(val_loader.dataset), len(calib_loader.dataset),
    )

    model = build_fm2_model(fm).to(dev)
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
            model, train_loader, cfg=fm,
            confinement_k=confinement_k, max_n_atoms=max_n_atoms,
            device=dev, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, train=True,
        )
        val_metrics = _epoch(
            model, val_loader, cfg=fm,
            confinement_k=confinement_k, max_n_atoms=max_n_atoms,
            device=dev, optimizer=None, scheduler=None,
            scaler=None, train=False,
        )
        logger.info(
            "  train: total={:.4f} huber={:.4f} nonneg={:.4f} mae={:.4f}",
            train_metrics["total"], train_metrics["huber"],
            train_metrics["nonneg"], train_metrics["mae"],
        )
        logger.info(
            "  val:   total={:.4f} huber={:.4f} nonneg={:.4f} mae={:.4f}",
            val_metrics["total"], val_metrics["huber"],
            val_metrics["nonneg"], val_metrics["mae"],
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

    from fmllm.fms.common import load_checkpoint
    from fmllm.fms.probe_runner import collect_items_from_loader, run_all_probes
    from fmllm.fms._schemas import save_probe_report

    load_checkpoint(best_path, model=model, map_location=dev)
    val_items = collect_items_from_loader(val_loader, n_items=128)
    probe_report = run_all_probes(
        fm.name, model=model, items=val_items, device=dev,
    )
    probe_report_path = out_dir / "probe_report.yaml"
    save_probe_report(probe_report, probe_report_path)
    logger.info(
        "Probe report saved to {} ({} probes)",
        probe_report_path, len(probe_report.results),
    )

    write_manifest(
        out_dir / "manifest.yaml",
        script="fmllm.fms.fm2_rdf.train",
        inputs={"h5_path": str(h5_path), "splits_path": str(splits_path)},
        config={
            "fm2": fm.model_dump(),
            "run_id": run_id,
            "device": str(dev),
            "epochs": epochs,
            "train_split": train_split,
        },
        extra={
            "n_params": n_params,
            "best_val_total": best_val,
            "split_sizes": {k: len(v) for k, v in sub.items()},
            "elapsed_seconds": elapsed,
            "history": history,
            "probe_report_path": str(probe_report_path),
            "probe_summary": [
                {"name": r.constraint_name, "score": r.satisfaction_score,
                 "passes": r.passes_threshold}
                for r in probe_report.results
            ],
        },
    )
    return best_path


__all__ = ["compute_fm2_losses", "train"]
