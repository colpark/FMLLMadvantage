"""CLI: Pipeline B trainer (SFT, DPO, or GRPO).

Reads a JSONL trajectory file, converts it into the appropriate
trainer-shaped dataset, runs the trainer, and saves the LoRA adapter
plus a manifest under the output directory.

Usage:
    # SFT on verifier-passing trajectories
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_pipeline_b.py \\
        --mode sft \\
        --trajectories runs/trajectories/<run_id>/trajectories.jsonl \\
        --out checkpoints/pipeline-b-sft

    # DPO on (PASS, FAIL) preference pairs
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_pipeline_b.py \\
        --mode dpo \\
        --trajectories runs/trajectories/<run_id>/trajectories.jsonl \\
        --out checkpoints/pipeline-b-dpo

    # GRPO with verifier reward
    accelerate launch --num_processes 4 scripts/train_pipeline_b.py \\
        --mode grpo \\
        --trajectories runs/trajectories/<run_id>/trajectories.jsonl \\
        --out checkpoints/pipeline-b-grpo

Depends on:
    typer, transformers, trl, peft, datasets, torch, loguru.
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.logging import configure_logging  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


class Mode(str, Enum):
    sft = "sft"
    dpo = "dpo"
    grpo = "grpo"


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    mode: Mode = typer.Option(..., "--mode"),
    trajectories: Path = typer.Option(..., "--trajectories"),
    out: Path = typer.Option(..., "--out"),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    base_model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "--base-model",
    ),
    learning_rate: float = typer.Option(0.0, "--learning-rate"),
    epochs: int = typer.Option(0, "--epochs"),
    lora_r: int = typer.Option(16, "--lora-r"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    bf16: bool = typer.Option(True, "--bf16"),
    seed: int = typer.Option(0, "--seed"),
    # GRPO-only
    num_generations: int = typer.Option(4, "--num-generations"),
    max_completion_length: int = typer.Option(1024, "--max-completion-length"),
    # Verifier wiring (GRPO only)
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    checkpoint_root: Path = typer.Option(Path("checkpoints"), "--checkpoint-root"),
    train_split: str = typer.Option("train_50k", "--train-split"),
    literature_db: Path = typer.Option(
        Path("data/literature/clusters.json"), "--literature-db",
    ),
) -> None:
    """Train Pipeline B with SFT, DPO, or GRPO."""
    cfg = load_config(config)

    run_id = generate_run_id(f"pipeline-b-{mode.value}")
    out_dir = out if out.is_absolute() else Path(out)
    out_dir = out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir)

    typer.echo(f"==> Pipeline B / {mode.value}")
    typer.echo(f"==> Run id: {run_id}")
    typer.echo(f"==> Output : {out_dir}")

    from fmllm.training import (
        load_trajectories_jsonl,
        trajectories_to_dpo_pairs,
        trajectories_to_grpo_prompts,
        trajectories_to_sft_records,
    )

    trajs = load_trajectories_jsonl(trajectories)
    typer.echo(f"==> Loaded {len(trajs)} trajectories")

    if mode is Mode.sft:
        from fmllm.training.sft_trainer import train_sft

        records = trajectories_to_sft_records(trajs, only_passing=True)
        if not records:
            raise typer.BadParameter("no PASS trajectories found for SFT")
        typer.echo(f"==> SFT records: {len(records)}")
        kwargs = dict(
            base_model_name=base_model,
            sft_records=records,
            output_dir=out_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            bf16=bf16,
            seed=seed,
        )
        if learning_rate > 0:
            kwargs["learning_rate"] = learning_rate
        if epochs > 0:
            kwargs["num_train_epochs"] = epochs
        train_sft(**kwargs)

    elif mode is Mode.dpo:
        from fmllm.training.dpo_alternative import train_dpo

        pairs = trajectories_to_dpo_pairs(trajs)
        if not pairs:
            raise typer.BadParameter(
                "no PASS / FAIL preference pairs found; "
                "collect more trajectories first",
            )
        typer.echo(f"==> DPO pairs: {len(pairs)}")
        kwargs = dict(
            base_model_name=base_model,
            dpo_pairs=pairs,
            output_dir=out_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            bf16=bf16,
            seed=seed,
        )
        if learning_rate > 0:
            kwargs["learning_rate"] = learning_rate
        if epochs > 0:
            kwargs["num_train_epochs"] = epochs
        train_dpo(**kwargs)

    else:  # grpo
        import torch

        from fmllm.bridges import load_fm_context  # noqa: F401  (used transitively)
        from fmllm.data.dataset import LJSpecimenDataset
        from fmllm.orchestrator import build_runners_from_checkpoints
        from fmllm.training.grpo_trainer import train_grpo
        from fmllm.training.reward import make_verifier_reward_fn
        from fmllm.verifier import build_default_verifier

        prompts = trajectories_to_grpo_prompts(trajs, deduplicate=True)
        typer.echo(f"==> GRPO prompts: {len(prompts)}")
        if not prompts:
            raise typer.BadParameter("no GRPO prompts found")

        dataset = LJSpecimenDataset(h5_path)
        runners = build_runners_from_checkpoints(
            checkpoint_root=checkpoint_root,
            train_split=train_split,
            dataset=dataset,
            cfg=cfg,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        verifier = build_default_verifier(literature_db_path=literature_db)
        reward_fn = make_verifier_reward_fn(
            verifier=verifier, runners=runners,
        )
        kwargs = dict(
            base_model_name=base_model,
            grpo_prompts=prompts,
            reward_fn=reward_fn,
            output_dir=out_dir,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            bf16=bf16,
            seed=seed,
        )
        if learning_rate > 0:
            kwargs["learning_rate"] = learning_rate
        if epochs > 0:
            kwargs["num_train_epochs"] = epochs
        train_grpo(**kwargs)
        dataset.close()

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.train_pipeline_b",
        inputs={
            "mode": mode.value,
            "trajectories": str(trajectories),
            "base_model": base_model,
        },
        config={
            "run_id": run_id,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate_override": learning_rate,
            "epochs_override": epochs,
        },
        extra={
            "n_trajectories": len(trajs),
            "out_dir": str(out_dir),
        },
    )
    typer.echo(f"==> Adapter saved under {out_dir}/adapter/")


if __name__ == "__main__":
    app()
