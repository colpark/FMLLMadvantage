"""CLI: train a Top-K sparse autoencoder on FM2 CLS embeddings.

Phase 13 Stage 0. Forwards every training specimen through the
frozen supervised FM2, takes the CLS embedding, and trains a
Top-K SAE to reconstruct it. The resulting features are the basis
for Stage 1's correlation-based labelling and Stage 2's
prompt-injection.

Output:

    checkpoints/sae/<run_id>/sae.pt
    checkpoints/sae/<run_id>/manifest.yaml
    checkpoints/sae/<run_id>/training.yaml

Usage:

    bash scripts/train_sae.sh
    uv run python scripts/train_sae.py --hidden-dim 1024 --k 32 --epochs 30

Depends on:
    typer, torch, h5py, pyyaml.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.representation.sae import build_topk_sae  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _latest_fm2(checkpoint_root: Path, train_split: str) -> Path:
    parent = checkpoint_root / "fm2_rdf" / train_split
    cands = sorted(parent.glob("*"), key=lambda p: p.name, reverse=True)
    cands = [c for c in cands if (c / "model.pt").exists()]
    if not cands:
        raise typer.BadParameter(f"no fm2_rdf checkpoint under {parent}")
    return cands[0]


def _load_specimen_ids(splits_path: Path, split_name: str) -> list[int]:
    with splits_path.open("r") as f:
        splits = yaml.safe_load(f)
    if split_name == "train":
        return [int(x) for x in splits.get("train", [])]
    sub = splits.get("train_subsets") or {}
    if split_name in sub:
        return [int(x) for x in sub[split_name]]
    raise typer.BadParameter(f"unknown split {split_name!r}")


def _extract_cls(
    *,
    model: torch.nn.Module,
    h5_path: Path,
    specimen_ids: list[int],
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    blocks: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f, torch.no_grad():
        for start in range(0, len(specimen_ids), batch_size):
            batch_ids = specimen_ids[start : start + batch_size]
            rdfs_np = np.stack(
                [np.asarray(f["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device).float()
            hidden = model.encode(rdfs)
            cls = hidden[:, 0, :].detach().cpu().numpy()
            blocks.append(cls)
    return np.concatenate(blocks, axis=0)


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    splits_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/splits.yaml"), "--splits-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    checkpoint_root: Path = typer.Option(
        Path("checkpoints"), "--checkpoint-root",
    ),
    train_split: str = typer.Option("train_50k", "--train-split"),
    n_specimens: int = typer.Option(
        20000, "--n-specimens",
        help="Number of specimens used to train the SAE. 20K is plenty;"
             " more buys diminishing returns on a 320-dim input.",
    ),
    out: Path = typer.Option(Path("checkpoints/sae"), "--out", "-o"),
    hidden_dim: int = typer.Option(1024, "--hidden-dim"),
    k: int = typer.Option(32, "--k", help="Top-K active latents per row."),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    lr: float = typer.Option(1.0e-3, "--lr"),
    weight_decay: float = typer.Option(0.0, "--weight-decay"),
    seed: int = typer.Option(0, "--seed"),
    log_every: int = typer.Option(50, "--log-every"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Train the SAE over FM2 CLS embeddings."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_id = generate_run_id(f"sae-fm2-{train_split}")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fm2_ckpt = _latest_fm2(checkpoint_root, train_split)
    typer.echo(f"==> Run id    : {run_id}")
    typer.echo(f"==> Output    : {out_dir}")
    typer.echo(f"==> FM2 ckpt  : {fm2_ckpt}")
    typer.echo(f"==> SAE config: hidden_dim={hidden_dim}, k={k}")

    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    pool = _load_specimen_ids(splits_path, train_split)
    if not pool:
        raise typer.BadParameter(f"empty split {train_split!r}")
    pool = pool[: max(n_specimens, 1)]
    typer.echo(f"==> Specimens : {len(pool)}")

    typer.echo("==> Extracting CLS embeddings...")
    cls = _extract_cls(
        model=fm2, h5_path=h5_path, specimen_ids=pool,
        device=device, batch_size=512,
    )
    in_dim = int(cls.shape[1])
    typer.echo(f"    cls shape: {cls.shape}")

    # Standardize features (zero mean, unit std). The SAE's pre_bias
    # absorbs the mean too, but normalizing helps the decoder
    # renormalization stay well-behaved.
    cls_mean = cls.mean(axis=0, keepdims=True)
    cls_std = cls.std(axis=0, keepdims=True).clip(min=1.0e-6)
    cls_norm = ((cls - cls_mean) / cls_std).astype(np.float32)

    sae = build_topk_sae(in_dim=in_dim, hidden_dim=hidden_dim, k=k).to(device)
    optimizer = torch.optim.AdamW(
        sae.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
    )

    typer.echo("")
    typer.echo("==> Training")
    typer.echo("-" * 64)
    history: list[dict] = []
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        perm = np.random.permutation(cls_norm.shape[0])
        for start in range(0, cls_norm.shape[0], batch_size):
            batch = cls_norm[perm[start : start + batch_size]]
            x = torch.from_numpy(batch).to(device)
            recon, z = sae(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae._renormalize_decoder()
            step += 1
            if step % log_every == 0 or step == 1:
                # active fraction = mean of (z > 0) per row, divided by
                # hidden_dim. Should be ~k/hidden_dim by construction.
                active_frac = float((z > 0).float().mean().item())
                history.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "active_frac": active_frac,
                })
                typer.echo(
                    f"  epoch={epoch:>3} step={step:>6} "
                    f"loss={loss.item():.6f} "
                    f"active_frac={active_frac:.4f} "
                    f"elapsed={time.time() - t0:.1f}s"
                )

    typer.echo("-" * 64)

    # Save the SAE plus the normalization stats so the loader can
    # apply the same transform at inference.
    save_path = out_dir / "sae.pt"
    torch.save(
        {
            "state_dict": {
                k: v.detach().cpu() for k, v in sae.state_dict().items()
            },
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "k": k,
            "cls_mean": cls_mean.astype(np.float32),
            "cls_std": cls_std.astype(np.float32),
            "fm2_checkpoint": str(fm2_ckpt),
            "epochs": epochs,
            "n_specimens": len(pool),
        },
        save_path,
    )
    typer.echo(f"==> Saved SAE: {save_path}")

    with (out_dir / "training.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "history": history,
                "final_loss": history[-1]["loss"] if history else None,
                "wall_clock_seconds": time.time() - t0,
            },
            f,
            sort_keys=False,
        )
    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.train_sae",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "fm2_checkpoint": str(fm2_ckpt),
            "train_split": train_split,
            "n_specimens": len(pool),
        },
        config={
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "k": k,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        extra={
            "trainable_parameters": int(sae.num_parameters()),
        },
    )


if __name__ == "__main__":
    app()
