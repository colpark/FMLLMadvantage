"""Rescore an old JSONL against the current HDF5 ground truth.

Why this exists: if the HDF5 was rebuilt (e.g. a field-name fix
landed), JSONL records written under the old HDF5 carry stale
``ground_truth`` values. Compare-time scoring reads those stale
values, so the original JSONL is no longer apples-to-apples with
freshly-generated records.

This script:

  1. Reads an input JSONL (e.g. cot_sft_sae records).
  2. For each record, looks up the FRESH ground truth from the
     supplied HDF5 by specimen_id.
  3. Recomputes ``is_correct`` and ``per_axis_correct`` against
     the fresh ground truth, using the same criterion as
     ``fmllm.materials.ground_truth.is_correct``.
  4. Writes a rescored JSONL with the original ``claim`` /
     ``raw_text`` / ``probe_outputs`` / etc. preserved but the
     ``ground_truth`` and correctness fields refreshed.

The LLM's claims are NOT regenerated -- this is a re-evaluation,
not a re-inference. If the LLM was trained on broken labels and
emits biased claims, the rescored JSONL will reflect that bias
honestly under the corrected ground truth.

Output:

    <input>.parent/<input>.stem.rescored.jsonl
    or --output PATH

Usage:

    bash scripts/materials/rescore_records.sh \\
        --input runs/materials/holdout/cot_sft_sae/<run>/records.jsonl

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


def _per_axis_correct(claim: dict, gt: dict) -> dict:
    """Mirrors 09b._per_axis_correct -- with sentinel guard on sg."""
    if not claim:
        return {
            "formation_energy": False,
            "e_above_hull": False,
            "is_stable": False,
            "band_gap_class": False,
            "space_group": False,
        }
    try:
        e_form_ok = (
            abs(float(claim.get("formation_energy", -999.0))
                - float(gt["formation_energy"])) <= 0.05
        )
    except (TypeError, ValueError, KeyError):
        e_form_ok = False
    try:
        e_hull_ok = (
            abs(float(claim.get("e_above_hull", 999.0))
                - float(gt["e_above_hull"])) <= 0.025
        )
    except (TypeError, ValueError, KeyError):
        e_hull_ok = False
    try:
        stable_ok = bool(claim.get("is_stable")) == bool(gt["is_stable"])
    except (TypeError, ValueError, KeyError):
        stable_ok = False
    try:
        bg_ok = (
            str(claim.get("band_gap_class", "")).lower()
            == str(gt["band_gap_class"]).lower()
        )
    except (TypeError, ValueError, KeyError):
        bg_ok = False
    try:
        claim_sg = int(claim.get("space_group", -1))
        gt_sg = int(gt["space_group"])
        sg_ok = claim_sg == gt_sg and claim_sg >= 1 and gt_sg >= 1
    except (TypeError, ValueError, KeyError):
        sg_ok = False
    return {
        "formation_energy": bool(e_form_ok),
        "e_above_hull": bool(e_hull_ok),
        "is_stable": bool(stable_ok),
        "band_gap_class": bool(bg_ok),
        "space_group": bool(sg_ok),
    }


def _serialize_truth(gt: dict) -> dict:
    out: dict = {}
    for k, v in gt.items():
        if isinstance(v, np.bool_):
            out[k] = bool(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = str(v)
    return out


@app.command()
def main(
    input_path: Path = typer.Option(..., "--input", "-i"),
    output_path: Path | None = typer.Option(
        None, "--output", "-o",
        help="Default: <input>.rescored.jsonl in same directory.",
    ),
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
) -> None:
    """Rescore an old materials records JSONL against current HDF5."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.ground_truth import is_correct, truth_dict  # noqa: PLC0415

    if not input_path.exists():
        raise typer.BadParameter(f"missing {input_path}")
    if not h5_path.exists():
        raise typer.BadParameter(f"missing {h5_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".rescored.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port: rescore records against current HDF5")
    typer.echo(f"    input  : {input_path}")
    typer.echo(f"    h5     : {h5_path}")
    typer.echo(f"    output : {output_path}")
    typer.echo("")

    counters = {
        "total": 0,
        "correct_before": 0,
        "correct_after": 0,
        "missing_in_h5": 0,
    }
    per_axis_before: dict[str, int] = {
        k: 0 for k in (
            "formation_energy", "e_above_hull", "is_stable",
            "band_gap_class", "space_group",
        )
    }
    per_axis_after: dict[str, int] = dict(per_axis_before)

    n_h5: int = 0
    with h5py.File(h5_path, "r") as h5, \
            input_path.open("r") as in_f, output_path.open("w") as out_f:
        n_h5 = int(h5.attrs.get("n_specimens", h5["material_id"].shape[0]))
        for line in in_f:
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

            old_gt = rec.get("ground_truth") or {}
            old_axes = rec.get("per_axis_correct")
            if old_axes is None:
                old_axes = _per_axis_correct(rec.get("claim") or {}, old_gt)
            old_correct = bool(rec.get("is_correct", False))

            fresh_gt = truth_dict(h5, sid)
            claim = rec.get("claim") or {}
            new_correct = is_correct(claim, fresh_gt) if claim else False
            new_axes = _per_axis_correct(claim, fresh_gt)

            counters["total"] += 1
            if old_correct:
                counters["correct_before"] += 1
            if new_correct:
                counters["correct_after"] += 1
            for axis in per_axis_before:
                if old_axes.get(axis):
                    per_axis_before[axis] += 1
                if new_axes.get(axis):
                    per_axis_after[axis] += 1

            rec["ground_truth"] = _serialize_truth(fresh_gt)
            rec["is_correct"] = bool(new_correct)
            rec["per_axis_correct"] = {k: bool(v) for k, v in new_axes.items()}
            rec["rescored_against_h5"] = str(h5_path)
            rec["rescored_utc"] = datetime.now(UTC).isoformat()
            out_f.write(json.dumps(rec) + "\n")

    n = max(counters["total"], 1)
    typer.echo(f"==> Rescored {counters['total']} records to {output_path}")
    if counters["missing_in_h5"]:
        typer.echo(
            f"    skipped {counters['missing_in_h5']} records with "
            f"missing/invalid specimen_id"
        )
    typer.echo("")
    typer.echo(
        f"    joint accuracy before : "
        f"{counters['correct_before']}/{n} "
        f"= {counters['correct_before'] / n:.4f}"
    )
    typer.echo(
        f"    joint accuracy after  : "
        f"{counters['correct_after']}/{n} "
        f"= {counters['correct_after'] / n:.4f}"
    )
    typer.echo("")
    typer.echo("    per-axis BEFORE -> AFTER:")
    for axis in per_axis_before:
        b, a = per_axis_before[axis], per_axis_after[axis]
        delta = (a - b) / n
        sign = "+" if delta >= 0 else ""
        typer.echo(
            f"      {axis:<20s}: {b / n:.4f} -> {a / n:.4f} "
            f"({sign}{delta * 100:.1f} pp)"
        )

    summary_path = output_path.with_suffix(".summary.yaml")
    with summary_path.open("w") as f:
        yaml.safe_dump(
            {
                "input_jsonl": str(input_path),
                "output_jsonl": str(output_path),
                "h5_path": str(h5_path),
                "rescored_utc": datetime.now(UTC).isoformat(),
                "counters": counters,
                "joint_before": float(counters["correct_before"] / n),
                "joint_after": float(counters["correct_after"] / n),
                "per_axis_before": {
                    k: float(v / n) for k, v in per_axis_before.items()
                },
                "per_axis_after": {
                    k: float(v / n) for k, v in per_axis_after.items()
                },
            },
            f,
            sort_keys=False,
        )
    typer.echo(f"==> Summary: {summary_path}")


if __name__ == "__main__":
    app()
