"""CLI: run the OHVD pipeline end-to-end on a single specimen.

The script loads each FM model from the latest checkpoint at the
requested training scale, builds the matching bridge per FM, opens
the dataset, builds the verifier (with literature DB and optional
calibrated cross-FM tolerance matrix), instantiates an LLM (real
Llama 3.1 by default; mock for smoke tests), and runs the
:class:`OHVDLoop` against a single specimen.

Output:
    ``runs/<run_id>/trajectory.json``
    ``runs/<run_id>/manifest.yaml``
    ``runs/<run_id>/run.log``

Usage examples:

    # Real run on the remote with Llama 3.1 8B
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_pipeline.py \\
        --specimen-id 42 --train-split train_50k

    # Mock smoke test (no LLM weights needed)
    uv run python scripts/run_pipeline.py --specimen-id 42 \\
        --mock-script scripts/mock_scripts/example.json

The mock script is a JSON list of strings; each string is the LLM's
response for one turn. Useful for end-to-end shape verification on
a CPU-only host.

Depends on:
    typer, torch, loguru, pyyaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.bridges import load_fm_context  # noqa: E402
from fmllm.data.dataset import LJSpecimenDataset  # noqa: E402
from fmllm.orchestrator import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    MockLLM,
    OHVDLoop,
    TransformersLLM,
    build_runners_from_checkpoints,
)
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.logging import configure_logging  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402
from fmllm.verifier import (  # noqa: E402
    PhysicalStateClaim,
    SourcesConfig,
    build_default_verifier,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_mock_responses(path: Path) -> list[str]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise typer.BadParameter(
            f"mock script {path} must be a JSON list of strings"
        )
    return list(data)


@app.command()
def main(
    specimen_id: int = typer.Option(..., "--specimen-id"),
    query: str = typer.Option(
        "Identify the specimen's atom count, motif, and temperature. "
        "Use FM tools to gather evidence, propose a claim, and commit when confident.",
        "--query",
    ),
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
    out: Path = typer.Option(Path("runs"), "--out", "-o"),
    step_budget: int = typer.Option(16, "--step-budget"),
    ablation: str = typer.Option(
        "V4", "--ablation",
        help="One of V0..V4. The verifier sources_config preset.",
    ),
    llm_model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "--llm-model",
    ),
    llm_temperature: float = typer.Option(0.2, "--llm-temperature"),
    adapter_path: Path | None = typer.Option(
        None, "--adapter-path",
        help="Optional path to a Pipeline B LoRA adapter. When set, "
        "loads the base LLM then stacks the adapter on top.",
    ),
    device: str = typer.Option("auto", "--device"),
    mock_script: Path | None = typer.Option(
        None, "--mock-script",
        help="Optional JSON file of scripted LLM responses. Skips the real LLM.",
    ),
) -> None:
    """Run Pipeline A end-to-end on one specimen."""
    cfg = load_config(config)
    run_id = generate_run_id(f"pipeline-a-{specimen_id}")
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)

    typer.echo(f"==> Run id: {run_id}")
    typer.echo(f"==> Output : {run_dir}")

    # Open the dataset (full HDF5).
    dataset = LJSpecimenDataset(h5_path)

    # Build runners from checkpoints.
    runners = build_runners_from_checkpoints(
        checkpoint_root=checkpoint_root,
        train_split=train_split,
        dataset=dataset,
        cfg=cfg,
        device=device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )

    # Build verifier.
    verifier = build_default_verifier(
        literature_db_path=literature_db,
    )

    # Build LLM.
    if mock_script is not None:
        llm = MockLLM(_load_mock_responses(mock_script))
    else:
        llm = TransformersLLM(
            model_name=llm_model,
            device=device,
            temperature=llm_temperature,
            adapter_path=str(adapter_path) if adapter_path is not None else None,
        )
        if adapter_path is not None:
            typer.echo(f"==> Pipeline B: stacking LoRA adapter from {adapter_path}")

    # Build loop.
    loop = OHVDLoop(
        llm=llm,
        verifier=verifier,
        runners=runners,
        max_steps=step_budget,
        sources_config=SourcesConfig.for_ablation(ablation),
    )

    trajectory = loop.run(query=query, specimen_id=specimen_id, run_id=run_id)

    traj_path = run_dir / "trajectory.json"
    with traj_path.open("w") as f:
        json.dump(json.loads(trajectory.model_dump_json()), f, indent=2, sort_keys=True)
    typer.echo(f"==> Trajectory: {traj_path}")
    typer.echo(f"==> Termination: {trajectory.termination.value}")
    if trajectory.final_claim is not None:
        typer.echo(f"==> Final claim: {trajectory.final_claim.model_dump()}")
    if trajectory.final_verdict is not None:
        typer.echo(
            f"==> Final aggregate decision: "
            f"{trajectory.final_verdict.aggregate_decision.value}"
        )

    write_manifest(
        run_dir / "manifest.yaml",
        script="scripts.run_pipeline",
        inputs={
            "specimen_id": specimen_id,
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "checkpoint_root": str(checkpoint_root),
            "train_split": train_split,
            "literature_db": str(literature_db),
        },
        config={
            "run_id": run_id,
            "step_budget": step_budget,
            "ablation": ablation,
            "llm_model": llm_model if mock_script is None else f"mock:{mock_script}",
            "device": device,
        },
        extra={
            "n_steps": len(trajectory.steps),
            "termination": trajectory.termination.value,
        },
    )

    dataset.close()


if __name__ == "__main__":
    app()
