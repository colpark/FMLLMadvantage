"""CLI: Phase 11 Stage 2 SFT on synthetic CoT records.

Loads the JSONL produced by ``scripts/build_cot_dataset.py`` and
hands it to :func:`fmllm.training.sft_trainer.train_sft`. Saves the
LoRA adapter under ``checkpoints/cot-sft/<run_id>/adapter/``.

Usage:

    bash scripts/train_cot_sft.sh
    uv run python scripts/train_cot_sft.py --epochs 3

Depends on:
    typer, pyyaml. Heavy training stack (transformers, peft, datasets)
    is lazy-imported by sft_trainer.train_sft.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.training.sft_trainer import train_sft  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _latest_dataset_jsonl() -> Path | None:
    cands = sorted(
        Path("runs/cot_datasets").glob("*/records.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


def _load_records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@app.command()
def main(
    dataset: Path | None = typer.Option(
        None, "--dataset",
        help="Path to records.jsonl. Default: latest under runs/cot_datasets/.",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    out: Path = typer.Option(
        Path("checkpoints/cot-sft"), "--out", "-o",
    ),
    epochs: int = typer.Option(3, "--epochs"),
    learning_rate: float = typer.Option(1.0e-4, "--learning-rate"),
    lora_r: int = typer.Option(16, "--lora-r"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    per_device_batch_size: int = typer.Option(1, "--per-device-batch-size"),
    grad_accum: int = typer.Option(16, "--grad-accum"),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    seed: int = typer.Option(0, "--seed"),
    run_id: str = typer.Option(
        "", "--run-id",
        help="Override the timestamp-based run id. Required for "
             "torchrun multi-GPU launches so every rank uses the "
             "same output directory.",
    ),
) -> None:
    """Run Phase 11 Stage 2 SFT on the synthetic CoT records."""
    if dataset is None:
        dataset = _latest_dataset_jsonl()
        if dataset is None:
            raise typer.BadParameter(
                "no records.jsonl under runs/cot_datasets/. "
                "Run scripts/build_cot_dataset.sh first."
            )
    if not dataset.exists():
        raise typer.BadParameter(f"dataset not found: {dataset}")

    # DDP-safe run_id: under torchrun, every rank executes this script
    # independently. To get a single shared output directory we accept
    # the run_id from the shell wrapper (--run-id) so all ranks see
    # the same value without any NCCL coordination. If not provided,
    # fall back to local generation (single-GPU path).
    import os as _os                     # noqa: PLC0415
    _local_rank_env = _os.environ.get("LOCAL_RANK")
    _world_size_env = int(_os.environ.get("WORLD_SIZE", "1"))
    _is_distributed = (
        _local_rank_env is not None
        and int(_local_rank_env) >= 0
        and _world_size_env > 1
    )

    if not run_id:
        run_id = generate_run_id("cot-sft-stage2")

    is_main = (not _is_distributed) or int(_local_rank_env) == 0

    out_dir = out / run_id
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> Dataset     : {dataset}")
    typer.echo(f"==> Base model  : {base_model}")
    typer.echo(f"==> Distributed : {_is_distributed} (is_main={is_main})")

    records = _load_records(dataset)
    typer.echo(f"==> Records     : {len(records)}")
    if not records:
        raise typer.BadParameter("dataset is empty")

    # train_sft consumes a list[dict] where each dict has a `messages`
    # field. Our records already do; pass them through unchanged.
    typer.echo("")
    typer.echo("==> Starting SFT (this may take a while)")
    train_sft(
        base_model_name=base_model,
        sft_records=records,
        output_dir=out_dir,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
    )

    if is_main:
        write_manifest(
            out_dir / "manifest.yaml",
            script="scripts.train_cot_sft",
            inputs={
                "dataset": str(dataset),
                "base_model": base_model,
                "n_records": len(records),
            },
            config={
                "run_id": run_id,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "per_device_batch_size": per_device_batch_size,
                "grad_accum": grad_accum,
                "max_seq_length": max_seq_length,
                "seed": seed,
                "world_size": _world_size_env,
            },
            extra={
                "stage": 2,
                "objective": "synthetic-cot-sft",
                "completed_utc": datetime.now(UTC).isoformat(),
            },
        )
        typer.echo(f"==> Adapter saved at: {out_dir / 'adapter'}")


if __name__ == "__main__":
    app()
