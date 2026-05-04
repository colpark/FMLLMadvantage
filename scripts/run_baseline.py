"""CLI: run one of the Phase 8a baselines on a range of specimens.

Three modes share the trajectory schema so the eight world-model
tests and the goal-accuracy metric score them uniformly:

* **naked** (B0): one-shot LLM commit, no FM tools, no verifier.
* **no_verifier** (B2): standard OHVD loop with NoOpVerifier
  always returning PASS. The LLM still calls FMs.
* **full** (B3): canonical Pipeline A.

Output (under ``runs/baselines/<baseline>/<run_id>/``):

    trajectories.jsonl    one Trajectory per line
    summary.yaml          counters by termination
    manifest.yaml         inputs and config

Usage:

    # B0 naked baseline on first 200 specimens
    uv run python scripts/run_baseline.py \\
        --baseline naked --start 0 --count 200

    # B2 no-verifier baseline on the same set
    uv run python scripts/run_baseline.py \\
        --baseline no_verifier --start 0 --count 200

    # B3 full pipeline (same as scripts/collect_trajectories.py)
    uv run python scripts/run_baseline.py \\
        --baseline full --start 0 --count 200

Depends on:
    typer, torch, pyyaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.baselines import NoOpVerifier, run_naked_baseline  # noqa: E402
from fmllm.data.dataset import LJSpecimenDataset  # noqa: E402
from fmllm.orchestrator import (  # noqa: E402
    MockLLM,
    TransformersLLM,
    build_runners_from_checkpoints,
)
from fmllm.training import collect_trajectories  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.logging import configure_logging  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
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
    return MockLLM(responses, cycle=True)


def _build_llm(
    *,
    mock_script: Path | None,
    llm_model: str,
    llm_temperature: float,
    adapter_path: Path | None,
    device: str,
) -> object:
    if mock_script is not None:
        return _load_mock(mock_script)
    return TransformersLLM(
        model_name=llm_model,
        device=device,
        temperature=llm_temperature,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
    )


@app.command()
def main(
    baseline: str = typer.Option(
        ..., "--baseline", "-b",
        help="One of: naked, no_verifier, full.",
    ),
    start: int = typer.Option(0, "--start"),
    count: int = typer.Option(200, "--count"),
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
    out: Path = typer.Option(Path("runs/baselines"), "--out", "-o"),
    max_steps: int = typer.Option(16, "--max-steps"),
    ablation: str = typer.Option("V4", "--ablation"),
    query: str = typer.Option(
        "Identify the specimen's atom count, motif, and temperature. "
        "Use FM tools to gather evidence, propose a claim, and commit when confident.",
        "--query",
    ),
    llm_model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "--llm-model",
    ),
    llm_temperature: float = typer.Option(0.4, "--llm-temperature"),
    adapter_path: Path | None = typer.Option(None, "--adapter-path"),
    device: str = typer.Option("auto", "--device"),
    mock_script: Path | None = typer.Option(None, "--mock-script"),
    specimen_ids_file: Path | None = typer.Option(
        None, "--specimen-ids-file",
        help="Optional JSON file with a list of specimen IDs to run, "
             "overriding --start/--count. Used by the held-out protocol.",
    ),
    literature_compare_energy: bool = typer.Option(
        False, "--literature-compare-energy/--no-literature-compare-energy",
        help="When true, the literature source raises CAVEAT on FM2-vs-"
             "ground-state energy mismatches. Default false because the "
             "DB references are ground-state and the data is finite-T. "
             "Only meaningful for --baseline full.",
    ),
) -> None:
    """Run one Phase 8a baseline."""
    if baseline not in {"naked", "no_verifier", "full"}:
        raise typer.BadParameter(
            f"--baseline must be one of naked, no_verifier, full; got {baseline!r}"
        )

    # Resolve the specimen ID list: explicit file overrides start/count.
    if specimen_ids_file is not None:
        if not specimen_ids_file.exists():
            raise typer.BadParameter(
                f"--specimen-ids-file not found: {specimen_ids_file}"
            )
        with specimen_ids_file.open("r") as f:
            specimen_ids = list(json.load(f))
        if not all(isinstance(x, int) for x in specimen_ids):
            raise typer.BadParameter(
                f"{specimen_ids_file} must contain a JSON list of ints"
            )
        run_id_slug = f"baseline-{baseline}-{len(specimen_ids)}-holdout"
    else:
        specimen_ids = list(range(start, start + count))
        run_id_slug = f"baseline-{baseline}-{count}"

    run_id = generate_run_id(run_id_slug)
    out_dir = out / baseline / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir)

    typer.echo(f"==> Baseline: {baseline}")
    typer.echo(f"==> Run id  : {run_id}")
    typer.echo(f"==> Output  : {out_dir}")
    if specimen_ids_file is not None:
        typer.echo(
            f"==> Specimens: from {specimen_ids_file} "
            f"({len(specimen_ids)} IDs, first 3: {specimen_ids[:3]})"
        )
    else:
        typer.echo(f"==> Specimens: [{start}, {start + count})")
    if baseline == "full" and literature_compare_energy:
        typer.echo("==> Literature source: compare_energy=True (strict mode)")

    llm = _build_llm(
        mock_script=mock_script,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        adapter_path=adapter_path,
        device=device,
    )

    if baseline == "naked":
        # No FM loading, no verifier, no dataset reads.
        jsonl_path = out_dir / "trajectories.jsonl"
        counters = {
            "total": 0,
            "committed": 0,
            "parse_failure": 0,
            "llm_error": 0,
        }
        with jsonl_path.open("w") as f:
            for sid in specimen_ids:
                traj = run_naked_baseline(
                    llm=llm,
                    query=query,
                    specimen_id=int(sid),
                )
                counters["total"] += 1
                counters[traj.termination.value] = (
                    counters.get(traj.termination.value, 0) + 1
                )
                f.write(traj.model_dump_json() + "\n")

        typer.echo(f"==> JSONL: {jsonl_path}")
        with (out_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"baseline": "naked", "counters": counters}, f)
        write_manifest(
            out_dir / "manifest.yaml",
            script="scripts.run_baseline",
            inputs={
                "baseline": "naked",
                "start": start,
                "count": count,
                "specimen_ids_file": (
                    str(specimen_ids_file) if specimen_ids_file is not None else None
                ),
                "n_specimens": len(specimen_ids),
            },
            config={
                "run_id": run_id,
                "llm_model": llm_model if mock_script is None else f"mock:{mock_script}",
                "llm_temperature": llm_temperature,
                "max_steps": 1,
            },
            extra={"counters": counters},
        )
        typer.echo(json.dumps(counters, indent=2))
        return

    # no_verifier and full both need the dataset, FMs, and the OHVD loop.
    cfg = load_config(config)
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

    if baseline == "no_verifier":
        verifier = NoOpVerifier()
    else:
        verifier = build_default_verifier(
            literature_db_path=literature_db,
            literature_compare_energy=literature_compare_energy,
        )

    summary = collect_trajectories(
        llm=llm,
        verifier=verifier,
        runners=runners,
        specimen_ids=specimen_ids,
        out_dir=out_dir,
        query=query,
        max_steps=max_steps,
        sources_config=SourcesConfig.for_ablation(ablation),
        filter_passing=False,
    )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.run_baseline",
        inputs={
            "baseline": baseline,
            "start": start,
            "count": count,
            "specimen_ids_file": (
                str(specimen_ids_file) if specimen_ids_file is not None else None
            ),
            "n_specimens": len(specimen_ids),
            "h5_path": str(h5_path),
            "checkpoint_root": str(checkpoint_root),
            "train_split": train_split,
            "literature_db": str(literature_db) if baseline == "full" else None,
            "literature_compare_energy": (
                literature_compare_energy if baseline == "full" else None
            ),
        },
        config={
            "run_id": run_id,
            "llm_model": llm_model if mock_script is None else f"mock:{mock_script}",
            "llm_temperature": llm_temperature,
            "max_steps": max_steps,
            "ablation": ablation if baseline == "full" else "n/a (no_verifier)",
        },
        extra={"counters": summary["counters"]},
    )

    typer.echo(json.dumps(summary["counters"], indent=2))
    typer.echo(f"==> JSONL: {summary['jsonl_path']}")
    dataset.close()


if __name__ == "__main__":
    app()
