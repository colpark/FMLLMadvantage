"""CLI: train a Top-K SAE on harvested Qwen residual activations.

Phase 15 Stage B. Loads the (N, hidden_dim) activations matrix
emitted by ``scripts/harvest_qwen_activations.py``, normalizes per
feature, and trains a Top-K sparse autoencoder. The resulting SAE is
the basis for Stage C labelling and Stage D activation steering.

Output:

    checkpoints/qwen_sae/<run_id>/sae.pt          # state_dict + stats
    checkpoints/qwen_sae/<run_id>/training.yaml   # loss / sparsity log
    checkpoints/qwen_sae/<run_id>/manifest.yaml

Usage:

    bash scripts/train_qwen_sae.sh

Hyperparameters that matter on Qwen-scale activations:

  - ``hidden_dim``: typically 4-8x the activation dim. For Qwen 2.5
    7B (3584-d residual) we default to 16384, ~4.5x.
  - ``k``: Top-K sparsity. 64 is a reasonable starting point on
    16K codebook; smaller is more aggressive but harder to train.
  - ``epochs``: 30 is enough for the activations to settle on a
    small dataset; scale up if N is large.

Depends on:
    typer, torch, numpy, pyyaml.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.representation.sae import build_topk_sae  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


@app.command()
def main(
    activations_dir: Path | None = typer.Option(
        None, "--activations-dir",
        help="Directory containing activations.npy. Default: latest "
             "under runs/qwen_activations/.",
    ),
    out: Path = typer.Option(
        Path("checkpoints/qwen_sae"), "--out", "-o",
    ),
    hidden_dim: int = typer.Option(
        16384, "--hidden-dim",
        help="SAE codebook size. ~4-8x the activation dim is usual.",
    ),
    k: int = typer.Option(
        64, "--k",
        help="Top-K active features per row.",
    ),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(128, "--batch-size"),
    lr: float = typer.Option(3.0e-4, "--lr"),
    weight_decay: float = typer.Option(0.0, "--weight-decay"),
    seed: int = typer.Option(0, "--seed"),
    log_every: int = typer.Option(50, "--log-every"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Train the Top-K SAE on harvested Qwen activations."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)

    if activations_dir is None:
        activations_dir = _latest_dir(Path("runs/qwen_activations"))
        if activations_dir is None:
            raise typer.BadParameter(
                "no harvested activations under runs/qwen_activations/. "
                "Run scripts/harvest_qwen_activations.sh first."
            )
    acts_path = activations_dir / "activations.npy"
    if not acts_path.exists():
        raise typer.BadParameter(f"missing {acts_path}")

    run_id = generate_run_id("qwen-sae")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> Activations : {acts_path}")

    acts = np.load(acts_path).astype(np.float32)
    if acts.ndim != 2:
        raise typer.BadParameter(
            f"expected (N, hidden_dim) activations, got {acts.shape}"
        )
    in_dim = int(acts.shape[1])
    typer.echo(f"    shape       : {acts.shape}")

    if acts.shape[0] < batch_size:
        typer.echo(
            f"WARNING: only {acts.shape[0]} rows; consider harvesting more "
            f"trajectories before training a high-quality SAE."
        )

    typer.echo(
        f"==> SAE config  : in_dim={in_dim} hidden_dim={hidden_dim} k={k}"
    )

    # Per-feature normalization (zero-mean, unit-variance). Same recipe
    # as the FM2 SAE; pre-bias absorbs the residual mean.
    cls_mean = acts.mean(axis=0, keepdims=True)
    cls_std = acts.std(axis=0, keepdims=True).clip(min=1.0e-6)
    acts_norm = ((acts - cls_mean) / cls_std).astype(np.float32)

    sae = build_topk_sae(in_dim=in_dim, hidden_dim=hidden_dim, k=k).to(device)
    optimizer = torch.optim.AdamW(
        sae.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
    )

    typer.echo("==> Training")
    typer.echo("-" * 64)
    history: list[dict] = []
    step = 0
    t0 = time.time()

    n = acts_norm.shape[0]
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch = acts_norm[perm[start : start + batch_size]]
            x = torch.from_numpy(batch).to(device)
            recon, z = sae(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae._renormalize_decoder()
            step += 1
            if step % log_every == 0 or step == 1:
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

    save_path = out_dir / "sae.pt"
    torch.save(
        {
            "state_dict": {
                k_: v.detach().cpu() for k_, v in sae.state_dict().items()
            },
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "k": k,
            "cls_mean": cls_mean.astype(np.float32),
            "cls_std": cls_std.astype(np.float32),
            "activations_dir": str(activations_dir),
            "epochs": epochs,
            "n_rows": int(n),
        },
        save_path,
    )
    typer.echo(f"==> Saved SAE   : {save_path}")

    with (out_dir / "training.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "completed_utc": datetime.now(UTC).isoformat(),
                "history": history,
                "final_loss": history[-1]["loss"] if history else None,
                "wall_clock_seconds": time.time() - t0,
            },
            f,
            sort_keys=False,
        )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.train_qwen_sae",
        inputs={
            "activations_dir": str(activations_dir),
        },
        config={
            "run_id": run_id,
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
            "n_rows": int(n),
            "trainable_parameters": int(sae.num_parameters()),
        },
    )


if __name__ == "__main__":
    app()
