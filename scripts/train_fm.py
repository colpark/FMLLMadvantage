"""Unified CLI for FM training and calibration.

Usage:
    # Train FM1 on GPU 0:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fm.py \\
        --fm fm1 --config configs/default.yaml \\
        --h5-path data/synthetic_lj_v1/specimens.h5 \\
        --splits-path data/synthetic_lj_v1/splits.yaml

    # Run all three FMs in parallel across GPUs 0, 1, 2:
    bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fm.py --fm fm1 ... &
             CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_fm.py --fm fm2 ... &
             CUDA_VISIBLE_DEVICES=2 uv run python scripts/train_fm.py --fm fm3 ... ; wait'

    # Conformal-calibrate after training:
    uv run python scripts/train_fm.py --fm fm1 \\
        --calibrate-only \\
        --checkpoint checkpoints/fm1_image/<run_id>/model.pt \\
        --h5-path ...

The script wires the per-FM ``train`` and ``calibrate`` functions into
a single Typer command. Each FM trains in roughly 1 to 3 hours on a
single H100 at default config.

Depends on:
    typer, torch, loguru.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

from fmllm.utils.config import load_config


class FMName(str, Enum):
    fm1 = "fm1"
    fm2 = "fm2"
    fm3 = "fm3"


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    fm: FMName = typer.Option(..., "--fm", help="Which FM to train or calibrate."),
    config: Path = typer.Option(
        Path("configs/default.yaml"), "--config", "-c",
        help="Path to the YAML config file.",
    ),
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
        help="Path to the dataset HDF5 file.",
    ),
    splits_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/splits.yaml"), "--splits-path",
        help="Path to the splits YAML.",
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", "-o",
        help="Override the checkpoint output directory.",
    ),
    device: str = typer.Option(
        "auto", "--device", "-d",
        help="Compute device. 'auto' picks cuda when available, else cpu.",
    ),
    epochs: int | None = typer.Option(
        None, "--epochs", "-e",
        help="Override epoch count from the config.",
    ),
    calibrate_only: bool = typer.Option(
        False, "--calibrate-only",
        help="Skip training. Run conformal calibration on an existing checkpoint.",
    ),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint",
        help="Required with --calibrate-only. Path to a trained model.pt.",
    ),
) -> None:
    """Dispatch to per-FM training or calibration."""
    cfg = load_config(config)

    if calibrate_only:
        if checkpoint is None:
            raise typer.BadParameter("--checkpoint is required with --calibrate-only")
        if fm is FMName.fm1:
            from fmllm.fms.fm1_image.conformal import calibrate as cal_fn
        elif fm is FMName.fm2:
            from fmllm.fms.fm2_rdf.conformal import calibrate as cal_fn
        else:
            from fmllm.fms.fm3_traj.conformal import calibrate as cal_fn
        out = cal_fn(
            cfg=cfg,
            checkpoint_path=checkpoint,
            h5_path=h5_path,
            splits_path=splits_path,
            device=device,
        )
        typer.echo(f"Calibration written to {out}")
        return

    if fm is FMName.fm1:
        from fmllm.fms.fm1_image.train import train as train_fn
    elif fm is FMName.fm2:
        from fmllm.fms.fm2_rdf.train import train as train_fn
    else:
        from fmllm.fms.fm3_traj.train import train as train_fn

    out = train_fn(
        cfg=cfg,
        h5_path=h5_path,
        splits_path=splits_path,
        out_dir=out_dir,
        device=device,
        epochs=epochs,
    )
    typer.echo(f"Best checkpoint at {out}")


if __name__ == "__main__":
    app()
