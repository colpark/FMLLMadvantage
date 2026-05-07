"""Stage 5: train a Top-K SAE on cached CHGNet embeddings.

Mirrors ``scripts/train_sae.py`` from the LJ pipeline but with the
materials embeddings path. Reuses
``fmllm.representation.sae.TopKSAE`` directly.

Output:

    checkpoints/materials/sae/<run_id>/sae.pt
    checkpoints/materials/sae/<run_id>/training.yaml
    checkpoints/materials/sae/<run_id>/manifest.yaml

Usage:

    bash scripts/materials/05_train_sae.sh

Depends on:
    typer, numpy, torch, pyyaml.
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


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _generate_run_id(slug: str = "sae") -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{slug}"


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


@app.command()
def main(
    embeddings_dir: Path | None = typer.Option(
        None, "--embeddings-dir",
        help="Directory with embeddings.npy. Default: latest under "
             "runs/materials/embeddings/.",
    ),
    out: Path = typer.Option(
        Path("checkpoints/materials/sae"), "--out", "-o",
    ),
    hidden_dim: int = typer.Option(1024, "--hidden-dim"),
    k: int = typer.Option(32, "--k"),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    lr: float = typer.Option(1.0e-3, "--lr"),
    weight_decay: float = typer.Option(0.0, "--weight-decay"),
    seed: int = typer.Option(0, "--seed"),
    log_every: int = typer.Option(50, "--log-every"),
    resample_every: int = typer.Option(
        0, "--resample-every",
        help="Resample dead features every N steps (Bricken et al. 2023). "
             "0 disables. Recommended ~500-2000 for materials port.",
    ),
    resample_window: int = typer.Option(
        0, "--resample-window",
        help="Lookback window in steps for the dead-feature counter. "
             "Defaults to resample_every if not set.",
    ),
    resample_threshold: int = typer.Option(
        0, "--resample-threshold",
        help="Features with <= this many fires in the window are dead.",
    ),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Train Top-K SAE on cached CHGNet embeddings."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    torch.manual_seed(seed)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.representation.sae import build_topk_sae  # noqa: PLC0415

    if embeddings_dir is None:
        cand = _latest_dir(Path("runs/materials/embeddings"))
        if cand is None:
            raise typer.BadParameter(
                "no embeddings under runs/materials/embeddings/. Run "
                "scripts/materials/03_encode.sh first."
            )
        embeddings_dir = cand
    emb_path = embeddings_dir / "embeddings.npy"
    if not emb_path.exists():
        raise typer.BadParameter(f"missing {emb_path}")

    embeddings = np.load(emb_path).astype(np.float32)
    in_dim = int(embeddings.shape[1])
    n = int(embeddings.shape[0])

    # Standardize features (zero mean, unit std). The SAE's pre-bias
    # absorbs the mean too; normalizing helps the decoder's
    # column-renormalization stay well-behaved.
    cls_mean = embeddings.mean(axis=0, keepdims=True)
    cls_std = embeddings.std(axis=0, keepdims=True).clip(min=1.0e-6)
    cls_norm = ((embeddings - cls_mean) / cls_std).astype(np.float32)

    run_id = _generate_run_id()
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 5: train SAE")
    typer.echo(f"    embeddings_dir : {embeddings_dir}")
    typer.echo(f"    n_specimens    : {n}")
    typer.echo(f"    in_dim         : {in_dim}")
    typer.echo(f"    hidden_dim     : {hidden_dim}")
    typer.echo(f"    k              : {k}")
    typer.echo(f"    epochs         : {epochs}")
    typer.echo(f"    out_dir        : {out_dir}")
    typer.echo("")

    sae = build_topk_sae(in_dim=in_dim, hidden_dim=hidden_dim, k=k).to(device)
    optimizer = torch.optim.AdamW(
        sae.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
    )

    typer.echo("==> Training")
    typer.echo(f"    resample_every : {resample_every} (0 disables)")
    if resample_every > 0:
        if resample_window <= 0:
            resample_window = resample_every
        typer.echo(f"    resample_window: {resample_window}")
        typer.echo(f"    resample_thresh: {resample_threshold}")
    typer.echo("-" * 64)
    history: list[dict] = []
    resample_log: list[dict] = []
    step = 0
    t0 = time.time()

    # Sliding-window per-feature fire counter.
    fire_counter = torch.zeros(hidden_dim, device=device, dtype=torch.long)
    # Stash the most recent batch so we can use it as the sampling pool
    # at the next resample point.
    last_batch_x: torch.Tensor | None = None

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch = cls_norm[perm[start : start + batch_size]]
            x = torch.from_numpy(batch).to(device)
            recon, z = sae(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae._renormalize_decoder()
            step += 1

            if resample_every > 0:
                fire_counter += (z > 0).sum(dim=0).long()
                last_batch_x = x.detach()
                if step % resample_every == 0:
                    n_resampled = sae.resample_dead_features(
                        activation_history=fire_counter,
                        x_batch_normalized=last_batch_x,
                        threshold=resample_threshold,
                        optimizer=optimizer,
                    )
                    if n_resampled > 0:
                        n_dead_before = int(
                            (fire_counter <= resample_threshold).sum().item()
                        )
                        typer.echo(
                            f"  [step {step}] resampled "
                            f"{n_resampled} dead features "
                            f"(<= {resample_threshold} fires in last "
                            f"{resample_window} steps)"
                        )
                        resample_log.append({
                            "step": step,
                            "n_resampled": int(n_resampled),
                            "n_dead_before": n_dead_before,
                        })
                    fire_counter.zero_()

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
            "embeddings_dir": str(embeddings_dir),
            "epochs": epochs,
            "n_specimens": n,
        },
        save_path,
    )
    typer.echo(f"==> Saved SAE: {save_path}")

    with (out_dir / "training.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "completed_utc": datetime.now(UTC).isoformat(),
                "history": history,
                "resample_log": resample_log,
                "resample_every": resample_every,
                "resample_window": resample_window,
                "resample_threshold": resample_threshold,
                "final_loss": history[-1]["loss"] if history else None,
                "wall_clock_seconds": time.time() - t0,
            },
            f,
            sort_keys=False,
        )

    with (out_dir / "manifest.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "embeddings_dir": str(embeddings_dir),
                "in_dim": in_dim,
                "hidden_dim": hidden_dim,
                "k": k,
                "epochs": epochs,
                "n_specimens": n,
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": seed,
            },
            f,
            sort_keys=False,
        )


if __name__ == "__main__":
    app()
