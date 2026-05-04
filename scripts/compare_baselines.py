"""CLI: side-by-side comparison of baseline evaluation reports.

Takes one or more ``runs/eval/<run_id>/report.yaml`` paths (one per
baseline), loads each :class:`EvaluationReport`, and prints a table
keyed by test name with one column per baseline. Also writes the
combined comparison to ``runs/comparisons/<run_id>/comparison.yaml``.

Two input shapes:

* ``--report KEY=PATH`` (repeat) — explicit per-baseline labels.
* ``--report PATH`` (repeat) — labels default to the report's run_id.

The script always reports the headline metric per test plus the
status (PASS / FAIL / SKIP) for each baseline; details (per-field
accuracy, per-FM coverage, etc.) land in the YAML for follow-up
analysis.

Usage:

    uv run python scripts/compare_baselines.py \\
        --report naked=runs/eval/<id-naked>/report.yaml \\
        --report no_verifier=runs/eval/<id-nv>/report.yaml \\
        --report full=runs/eval/<id-full>/report.yaml

Depends on:
    typer, pyyaml.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _parse_report_arg(arg: str) -> tuple[str, Path]:
    if "=" in arg:
        key, path = arg.split("=", 1)
        return key.strip(), Path(path.strip())
    p = Path(arg.strip())
    return p.parent.name, p


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _flatten_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map test_name → result dict across all four buckets."""
    out: dict[str, dict[str, Any]] = {}
    for bucket in (
        "trajectory_results",
        "prediction_results",
        "cross_layer_results",
        "accuracy_results",
    ):
        for r in report.get(bucket) or []:
            out[r["test_name"]] = r
    return out


def _status(r: dict[str, Any]) -> str:
    if r.get("skipped"):
        return "SKIP"
    return "PASS" if r.get("passes") else "FAIL"


def _value(r: dict[str, Any]) -> str:
    v = r.get("metric_value")
    if v is None:
        return "n/a"
    return f"{float(v):.4f}"


@app.command()
def main(
    report: list[str] = typer.Option(
        ..., "--report", "-r",
        help="KEY=PATH or PATH; repeat per baseline.",
    ),
    out: Path = typer.Option(Path("runs/comparisons"), "--out", "-o"),
) -> None:
    """Compare baseline evaluation reports."""
    if len(report) < 2:
        raise typer.BadParameter("Provide at least two --report values.")

    parsed: list[tuple[str, Path, dict[str, Any]]] = []
    seen_names: list[str] = []
    for arg in report:
        key, path = _parse_report_arg(arg)
        if not path.exists():
            raise typer.BadParameter(f"report not found: {path}")
        rep = _load_report(path)
        # Disambiguate duplicate keys.
        original_key = key
        suffix = 1
        while key in seen_names:
            suffix += 1
            key = f"{original_key}_{suffix}"
        seen_names.append(key)
        parsed.append((key, path, rep))

    # Build the comparison table: rows = test_name, columns = baselines.
    by_baseline: dict[str, dict[str, dict[str, Any]]] = {
        key: _flatten_results(rep) for key, _, rep in parsed
    }
    test_order = []
    for key, _, rep in parsed:
        for bucket in (
            "trajectory_results",
            "prediction_results",
            "cross_layer_results",
            "accuracy_results",
        ):
            for r in rep.get(bucket) or []:
                if r["test_name"] not in test_order:
                    test_order.append(r["test_name"])

    run_id = generate_run_id("baseline-compare")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Comparison run id: {run_id}")
    typer.echo(f"==> Output : {out_dir}")
    typer.echo("")

    name_w = max(len("test"), max(len(t) for t in test_order))
    metric_w = 13
    status_w = 6
    header_cells = [f"{'test':<{name_w}}"]
    for key, _, _ in parsed:
        header_cells.append(f"{key[:metric_w]:<{metric_w}}")
        header_cells.append(f"{'':<{status_w}}")
    line_w = sum(len(c) + 1 for c in header_cells)

    typer.echo(" ".join(header_cells))
    typer.echo("-" * line_w)

    for test_name in test_order:
        row = [f"{test_name:<{name_w}}"]
        for key, _, _ in parsed:
            r = by_baseline[key].get(test_name)
            if r is None:
                row.append(f"{'-':<{metric_w}}")
                row.append(f"{'-':<{status_w}}")
            else:
                row.append(f"{_value(r):<{metric_w}}")
                row.append(f"{_status(r):<{status_w}}")
        typer.echo(" ".join(row))

    typer.echo("-" * line_w)

    # Headline accuracy line: compound goal accuracy per baseline.
    headline = ["accuracy"]
    for key, _, _ in parsed:
        r = by_baseline[key].get("goal_accuracy")
        v = "n/a" if r is None or r.get("metric_value") is None else f"{r['metric_value']:.3f}"
        headline.append(f"{key}={v}")
    typer.echo("HEADLINE: " + " | ".join(headline[1:]))

    out_yaml = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "baselines": [
            {"name": key, "report_path": str(path)}
            for key, path, _ in parsed
        ],
        "rows": [
            {
                "test_name": test_name,
                "values": {
                    key: {
                        "metric_value": (by_baseline[key].get(test_name) or {}).get("metric_value"),
                        "passes": (by_baseline[key].get(test_name) or {}).get("passes"),
                        "skipped": (by_baseline[key].get(test_name) or {}).get("skipped"),
                        "threshold": (by_baseline[key].get(test_name) or {}).get("threshold"),
                        "threshold_direction": (by_baseline[key].get(test_name) or {}).get("threshold_direction"),
                        "details": (by_baseline[key].get(test_name) or {}).get("details"),
                    }
                    for key, _, _ in parsed
                },
            }
            for test_name in test_order
        ],
    }
    out_path = out_dir / "comparison.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)
    typer.echo(f"==> Comparison written to {out_path}")


if __name__ == "__main__":
    app()
