"""CLI: run the eight world-model evaluation tests on collected trajectories.

Two input shapes are supported:

* **Single ablation** (default): pass ``--trajectories <path>`` for one
  trajectories.jsonl. Every test runs over the full set, and
  ``federated_factorability`` is skipped (only one ablation present).

* **Ablation lattice**: pass ``--ablation V0=path0 --ablation V1=path1 ...``
  to feed multiple ablations at once. The factorability test now has
  the data it needs. Every other test runs on the *union* of all
  ablations' trajectories.

The script writes ``runs/eval/<run_id>/report.yaml`` with an
:class:`EvaluationReport` plus a manifest. A short summary prints to
stdout: per-test metric, threshold, pass/fail, and the aggregate pass
flag.

Usage:

    # single set of trajectories
    uv run python scripts/run_evaluation.py \\
        --trajectories runs/trajectories/<id>/trajectories.jsonl \\
        --h5-path data/synthetic_lj_v1/specimens.h5

    # ablation lattice
    uv run python scripts/run_evaluation.py \\
        --ablation V0=runs/trajectories/V0/trajectories.jsonl \\
        --ablation V4=runs/trajectories/V4/trajectories.jsonl \\
        --h5-path data/synthetic_lj_v1/specimens.h5

Depends on:
    typer, pyyaml.
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

from fmllm.evaluation import (  # noqa: E402
    EvaluationReport,
    calibrated_uncertainty,
    federated_factorability,
    goal_competence,
    prediction_compression,
    prediction_distinction,
    step_recoverability,
    trajectory_compression,
    trajectory_distinction,
)
from fmllm.evaluation.utils import load_ground_truth  # noqa: E402
from fmllm.orchestrator import Trajectory  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_trajectories(path: Path) -> list[Trajectory]:
    """Read a JSONL file of trajectories (one per line)."""
    out: list[Trajectory] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Trajectory.model_validate_json(line))
    return out


def _parse_ablation_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise typer.BadParameter(
            f"--ablation expects KEY=PATH, got {arg!r}"
        )
    key, path = arg.split("=", 1)
    return key.strip(), Path(path.strip())


def _run_all_tests(
    trajectories: list[Trajectory],
    truth: dict[int, dict[str, Any]],
    trajectories_by_ablation: dict[str, list[Trajectory]] | None,
) -> tuple[list, list, list]:
    traj_results = [
        trajectory_compression.measure(trajectories=trajectories, truth=truth),
        trajectory_distinction.measure(trajectories=trajectories, truth=truth),
        step_recoverability.measure(trajectories=trajectories),
    ]
    pred_results = [
        prediction_compression.measure(trajectories=trajectories, truth=truth),
        prediction_distinction.measure(trajectories=trajectories, truth=truth),
        goal_competence.measure(trajectories=trajectories, truth=truth),
    ]
    cross_results = []
    if trajectories_by_ablation is not None and len(trajectories_by_ablation) >= 2:
        cross_results.append(
            federated_factorability.measure(
                trajectories_by_ablation=trajectories_by_ablation,
            )
        )
    else:
        from fmllm.evaluation.schema import make_skipped
        cross_results.append(
            make_skipped(
                test_name="federated_factorability",
                layer="cross_layer",
                metric_name="factorability_score",
                threshold=0.45,
                threshold_direction="ge",
                reason="single ablation; cannot compare presets",
            )
        )
    cross_results.append(
        calibrated_uncertainty.measure(trajectories=trajectories),
    )
    return traj_results, pred_results, cross_results


@app.command()
def main(
    trajectories: Path | None = typer.Option(
        None, "--trajectories", "-t",
        help="Path to a single trajectories.jsonl. Mutually exclusive with --ablation.",
    ),
    ablation: list[str] = typer.Option(
        [], "--ablation", "-a",
        help="KEY=PATH form, can be repeated. Use V0..V4 keys for the factorability test.",
    ),
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    out: Path = typer.Option(Path("runs/eval"), "--out", "-o"),
    fail_on_error: bool = typer.Option(
        False, "--fail-on-error",
        help="Exit non-zero when any test does not pass (skipped tests excluded).",
    ),
) -> None:
    """Run all eight world-model evaluation tests."""
    if trajectories is None and not ablation:
        raise typer.BadParameter(
            "Provide --trajectories <path> OR one or more --ablation KEY=PATH."
        )
    if trajectories is not None and ablation:
        raise typer.BadParameter(
            "Pass either --trajectories or --ablation, not both."
        )

    if trajectories is not None:
        traj_paths = {"single": trajectories}
        trajectories_by_ablation = None
        all_trajectories = _load_trajectories(trajectories)
    else:
        traj_paths = {}
        trajectories_by_ablation = {}
        all_trajectories = []
        for arg in ablation:
            key, path = _parse_ablation_arg(arg)
            traj_paths[key] = path
            trajs = _load_trajectories(path)
            trajectories_by_ablation[key] = trajs
            all_trajectories.extend(trajs)

    # Load ground truth only for specimens we actually saw.
    seen_ids = sorted({
        int(t.specimen_id) for t in all_trajectories if t.specimen_id is not None
    })
    if not seen_ids:
        typer.echo("ERROR: no trajectories carry a specimen_id; nothing to evaluate.")
        raise typer.Exit(code=1)
    typer.echo(f"==> Loaded {len(all_trajectories)} trajectories across {len(seen_ids)} specimens")
    truth = load_ground_truth(h5_path, specimen_ids=seen_ids)

    run_id = generate_run_id("evaluation")
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"==> Run id: {run_id}")
    typer.echo(f"==> Output : {run_dir}")

    traj_results, pred_results, cross_results = _run_all_tests(
        trajectories=all_trajectories,
        truth=truth,
        trajectories_by_ablation=trajectories_by_ablation,
    )

    all_results = traj_results + pred_results + cross_results
    n_pass = sum(1 for r in all_results if r.passes and not r.skipped)
    n_skip = sum(1 for r in all_results if r.skipped)
    n_fail = sum(1 for r in all_results if not r.passes and not r.skipped)
    aggregate_pass = (n_fail == 0) and (n_skip < len(all_results))

    report = EvaluationReport(
        run_id=run_id,
        timestamp_utc=datetime.now(UTC).isoformat(),
        trajectory_results=traj_results,
        prediction_results=pred_results,
        cross_layer_results=cross_results,
        aggregate_pass=aggregate_pass,
        inputs={
            "h5_path": str(h5_path),
            "trajectory_paths": {k: str(v) for k, v in traj_paths.items()},
            "n_trajectories": len(all_trajectories),
            "n_specimens": len(seen_ids),
        },
    )

    report_path = run_dir / "report.yaml"
    with report_path.open("w") as f:
        yaml.safe_dump(json.loads(report.model_dump_json()), f, sort_keys=False)
    typer.echo(f"==> Report: {report_path}")

    write_manifest(
        run_dir / "manifest.yaml",
        script="scripts.run_evaluation",
        inputs={
            "h5_path": str(h5_path),
            "trajectory_paths": {k: str(v) for k, v in traj_paths.items()},
        },
        config={
            "run_id": run_id,
        },
        extra={
            "n_trajectories": len(all_trajectories),
            "n_specimens": len(seen_ids),
            "n_tests": len(all_results),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_skip": n_skip,
            "aggregate_pass": aggregate_pass,
        },
    )

    typer.echo("")
    typer.echo("Results")
    typer.echo("-" * 96)
    typer.echo(
        f"{'test':<28} {'layer':<14} {'metric':<14} {'threshold':<11} {'status':<10} samples"
    )
    typer.echo("-" * 96)
    for r in all_results:
        status = "SKIP" if r.skipped else ("PASS" if r.passes else "FAIL")
        metric = "n/a" if r.metric_value is None else f"{r.metric_value:.4f}"
        thresh = "n/a" if r.threshold is None else f"{r.threshold_direction} {r.threshold:.3f}"
        typer.echo(
            f"{r.test_name:<28} {r.layer:<14} {metric:<14} {thresh:<11} {status:<10} {r.n_samples}"
        )
    typer.echo("-" * 96)
    typer.echo(
        f"AGGREGATE: pass={n_pass}, fail={n_fail}, skip={n_skip}, "
        f"all-pass={'yes' if aggregate_pass else 'no'}"
    )

    if fail_on_error and n_fail > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
