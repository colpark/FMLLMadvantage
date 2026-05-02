"""CLI: collect Pipeline A trajectories on a list of specimens.

Loads each FM model from the latest checkpoint, builds bridges and
verifier, and runs the OHVD loop on every specimen in a configurable
range. Writes JSONL plus a summary.

Usage:
    # Collect trajectories on the first 1000 train_50k specimens
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/collect_trajectories.py \\
        --start 0 --count 1000 \\
        --train-split train_50k \\
        --out runs/trajectories/pipeline-a-50k

    # Collect with the mock LLM (no LLM weights needed)
    uv run python scripts/collect_trajectories.py \\
        --start 0 --count 200 \\
        --mock-script scripts/mock_scripts/example.json \\
        --out runs/trajectories/mock-smoke

Depends on:
    typer, torch, loguru.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.bridges import load_fm_context  # noqa: E402
from fmllm.data.dataset import LJSpecimenDataset  # noqa: E402
from fmllm.orchestrator import (  # noqa: E402
    MockLLM, TransformersLLM, build_runners_from_checkpoints,
)
from fmllm.training import collect_trajectories  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.logging import configure_logging  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402
from fmllm.verifier import SourcesConfig, build_default_verifier  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_mock(path: Path) -> MockLLM:
    with path.open("r") as f:
        responses = json.load(f)
    if not isinstance(responses, list) or not all(isinstance(x, str) for x in responses):
        raise typer.BadParameter(
            f"mock script {path} must be a JSON list of strings",
        )
    return MockLLM(responses)


@app.command()
def main(
    start: int = typer.Option(0, "--start"),
    count: int = typer.Option(100, "--count"),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    splits_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/splits.yaml"), "--splits-path",
    ),
    checkpoint_root: Path = typer.Option(Path("checkpoints"), "--checkpoint-root"),
    train_split: str = typer.Option("train_50k", "--train-split"),
    literature_db: Path = typer.Option(
        Path("data/literature/clusters.json"), "--literature-db",
    ),
    out: Path = typer.Option(Path("runs/trajectories"), "--out", "-o"),
    max_steps: int = typer.Option(16, "--max-steps"),
    ablation: str = typer.Option("V4", "--ablation"),
    filter_passing: bool = typer.Option(False, "--filter-passing"),
    llm_model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "--llm-model",
    ),
    llm_temperature: float = typer.Option(0.4, "--llm-temperature"),
    device: str = typer.Option("auto", "--device"),
    mock_script: Path | None = typer.Option(None, "--mock-script"),
) -> None:
    """Run Pipeline A across a range of specimens and write JSONL."""
    cfg = load_config(config)

    run_id = generate_run_id(f"traj-{train_split}-{count}")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir)

    typer.echo(f"==> Run id: {run_id}")
    typer.echo(f"==> Output : {out_dir}")
    typer.echo(f"==> Specimens: [{start}, {start + count})")

    dataset = LJSpecimenDataset(h5_path)
    runners = build_runners_from_checkpoints(
        checkpoint_root=checkpoint_root,
        train_split=train_split,
        dataset=dataset,
        cfg=cfg,
        device=device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    verifier = build_default_verifier(literature_db_path=literature_db)

    if mock_script is not None:
        llm = _load_mock(mock_script)
    else:
        llm = TransformersLLM(
            model_name=llm_model,
            device=device,
            temperature=llm_temperature,
        )

    summary = collect_trajectories(
        llm=llm,
        verifier=verifier,
        runners=runners,
        specimen_ids=range(start, start + count),
        out_dir=out_dir,
        max_steps=max_steps,
        sources_config=SourcesConfig.for_ablation(ablation),
        filter_passing=filter_passing,
    )

    typer.echo(json.dumps(summary["counters"], indent=2))
    typer.echo(f"==> JSONL: {summary['jsonl_path']}")
    dataset.close()


if __name__ == "__main__":
    app()
