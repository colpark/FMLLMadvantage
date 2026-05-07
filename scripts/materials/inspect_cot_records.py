"""Diagnostic: check whether CoT-record ground_truth matches current HDF5.

For each record in a CoT dataset JSONL, look up the FRESH ground
truth from the HDF5 by specimen_id and compare against the record's
stored ``ground_truth`` field. Reports per-axis consistency and
shows the first N disagreements side-by-side.

Use this after rebuilding the HDF5 (e.g. fixed a field-name bug)
to decide whether the CoT records need rebuilding too. Records
whose stored ground_truth disagrees with fresh HDF5 will have
trained the SFT adapter on incorrect labels for those axes.

Output:

    runs/materials/diagnostics/<run_id>/cot_records_consistency.yaml

Usage:

    bash scripts/materials/inspect_cot_records.sh
    bash scripts/materials/inspect_cot_records.sh --records-path <jsonl>

Depends on:
    typer, h5py, numpy, pyyaml.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _latest_records(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*/records.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


_AXES_CONTINUOUS = ("formation_energy", "e_above_hull", "band_gap")
_AXES_CATEGORICAL = (
    "is_stable", "is_metal", "band_gap_class", "crystal_system",
    "space_group", "n_atoms",
)


def _close(a, b, tol: float = 1.0e-3) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _equal(a, b) -> bool:
    """Type-tolerant equality for categorical fields."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        if isinstance(a, str) or isinstance(b, str):
            return str(a).strip().lower() == str(b).strip().lower()
        return a == b
    except Exception:
        return False


def _format_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


@app.command()
def main(
    records_path: Path | None = typer.Option(
        None, "--records-path",
        help="Path to records.jsonl. Default: latest under "
             "runs/materials/cot_datasets_sae/.",
    ),
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    n_show: int = typer.Option(
        10, "--n-show",
        help="How many side-by-side disagreements to show.",
    ),
    out: Path = typer.Option(
        Path("runs/materials/diagnostics"), "--out", "-o",
    ),
) -> None:
    """Compare CoT-record ground_truth to current HDF5 truth."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.ground_truth import truth_dict  # noqa: PLC0415

    if records_path is None:
        records_path = _latest_records(Path("runs/materials/cot_datasets_sae"))
        if records_path is None:
            raise typer.BadParameter(
                "no records.jsonl under runs/materials/cot_datasets_sae/. "
                "Run scripts/materials/07_build_cot.sh first or pass "
                "--records-path explicitly."
            )
    if not h5_path.exists():
        raise typer.BadParameter(f"missing {h5_path}")

    run_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-cot-records-check"
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> CoT records consistency check")
    typer.echo(f"    records : {records_path}")
    typer.echo(f"    h5      : {h5_path}")
    typer.echo("")

    counters: dict[str, int] = {"total": 0, "missing_in_h5": 0}
    axis_match: dict[str, int] = {
        ax: 0 for ax in (_AXES_CONTINUOUS + _AXES_CATEGORICAL)
    }
    axis_present: dict[str, int] = dict(axis_match)
    disagreements: dict[str, list[dict]] = {
        ax: [] for ax in (_AXES_CONTINUOUS + _AXES_CATEGORICAL)
    }
    n_h5: int = 0

    with h5py.File(h5_path, "r") as h5, records_path.open("r") as f:
        n_h5 = int(h5.attrs.get("n_specimens", h5["material_id"].shape[0]))
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("specimen_id")
            if not isinstance(sid, int) or not (0 <= sid < n_h5):
                counters["missing_in_h5"] += 1
                continue

            stored = rec.get("ground_truth") or {}
            fresh = truth_dict(h5, sid)
            counters["total"] += 1

            for ax in _AXES_CONTINUOUS:
                if ax not in stored or ax not in fresh:
                    continue
                axis_present[ax] += 1
                if _close(stored[ax], fresh[ax]):
                    axis_match[ax] += 1
                elif len(disagreements[ax]) < n_show:
                    disagreements[ax].append({
                        "sid": int(sid),
                        "stored": stored[ax],
                        "fresh": fresh[ax],
                    })

            for ax in _AXES_CATEGORICAL:
                if ax not in stored or ax not in fresh:
                    continue
                axis_present[ax] += 1
                if _equal(stored[ax], fresh[ax]):
                    axis_match[ax] += 1
                elif len(disagreements[ax]) < n_show:
                    disagreements[ax].append({
                        "sid": int(sid),
                        "stored": stored[ax],
                        "fresh": fresh[ax],
                    })

    n = max(counters["total"], 1)
    typer.echo(f"==> Examined {counters['total']} records")
    if counters["missing_in_h5"]:
        typer.echo(
            f"    skipped {counters['missing_in_h5']} records "
            f"with invalid specimen_id"
        )
    typer.echo("")

    typer.echo("Per-axis consistency (record's stored vs fresh HDF5):")
    typer.echo(f"  {'axis':<22s} {'present':>10s} {'match':>10s} {'rate':>10s}")
    issue_axes: list[str] = []
    for ax in (_AXES_CONTINUOUS + _AXES_CATEGORICAL):
        present = axis_present[ax]
        match = axis_match[ax]
        if present == 0:
            typer.echo(f"  {ax:<22s} {'-':>10s} {'-':>10s} {'(absent)':>10s}")
            continue
        rate = match / present
        flag = " <-- DRIFT" if rate < 0.99 else ""
        typer.echo(
            f"  {ax:<22s} {present:>10d} {match:>10d} {rate:>9.4f}{flag}"
        )
        if rate < 0.99:
            issue_axes.append(ax)
    typer.echo("")

    if issue_axes:
        typer.echo("Side-by-side disagreement examples:")
        for ax in issue_axes:
            ex = disagreements[ax]
            if not ex:
                continue
            typer.echo(f"  [{ax}]")
            for e in ex[:n_show]:
                typer.echo(
                    f"    sid={e['sid']:<8d} "
                    f"stored={_format_value(e['stored'])}  "
                    f"fresh={_format_value(e['fresh'])}"
                )
        typer.echo("")
        typer.echo(
            f"VERDICT: {len(issue_axes)} axes drifted vs current HDF5. "
            f"Records were built against an older/different HDF5; rebuild "
            f"recommended (scripts/materials/07_build_cot.sh) before "
            f"trusting any SFT trained on these records."
        )
    else:
        typer.echo(
            "VERDICT: every axis matches current HDF5. Records are "
            "consistent; no rebuild needed."
        )

    summary = {
        "records_path": str(records_path),
        "h5_path": str(h5_path),
        "examined_utc": datetime.now(UTC).isoformat(),
        "counters": counters,
        "axis_present": axis_present,
        "axis_match": axis_match,
        "axis_match_rate": {
            ax: float(axis_match[ax] / axis_present[ax])
            if axis_present[ax] else None
            for ax in (_AXES_CONTINUOUS + _AXES_CATEGORICAL)
        },
        "drifted_axes": issue_axes,
        "disagreement_examples": disagreements,
    }
    with (out_dir / "cot_records_consistency.yaml").open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    typer.echo(f"==> Report: {out_dir / 'cot_records_consistency.yaml'}")


if __name__ == "__main__":
    app()
