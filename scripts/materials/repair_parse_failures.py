"""Re-parse raw_text in an existing records.jsonl with the lenient parser.

For every record whose ``claim`` is None, try the lenient
parser on ``raw_text``. If a claim is recoverable, write the
recovered claim, recompute ``is_correct`` and ``per_axis_correct``
against the record's ground_truth, and update parse_failure
counters. Records that already parsed are passed through
unchanged.

Use this after upgrading the parser (or fixing a bug in it) to
recover already-emitted Stage 9 outputs without re-running
inference (~3 min saved per holdout).

Output:

    <input>.parent/<input>.stem.repaired.jsonl
    or --output PATH

Usage:

    bash scripts/materials/repair_parse_failures.sh
    bash scripts/materials/repair_parse_failures.sh --input <jsonl>

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


def _per_axis_correct(claim: dict, gt: dict) -> dict:
    """Same per-axis criterion as 09b._per_axis_correct."""
    if not claim:
        return {ax: False for ax in (
            "formation_energy", "e_above_hull", "is_stable",
            "band_gap_class", "space_group",
        )}
    out: dict = {}
    try:
        out["formation_energy"] = (
            abs(float(claim.get("formation_energy", -999.0))
                - float(gt["formation_energy"])) <= 0.05
        )
    except (TypeError, ValueError, KeyError):
        out["formation_energy"] = False
    try:
        out["e_above_hull"] = (
            abs(float(claim.get("e_above_hull", 999.0))
                - float(gt["e_above_hull"])) <= 0.025
        )
    except (TypeError, ValueError, KeyError):
        out["e_above_hull"] = False
    try:
        out["is_stable"] = bool(claim.get("is_stable")) == bool(gt["is_stable"])
    except (TypeError, ValueError, KeyError):
        out["is_stable"] = False
    try:
        out["band_gap_class"] = (
            str(claim.get("band_gap_class", "")).lower()
            == str(gt["band_gap_class"]).lower()
        )
    except (TypeError, ValueError, KeyError):
        out["band_gap_class"] = False
    try:
        claim_sg = int(claim.get("space_group", -1))
        gt_sg = int(gt["space_group"])
        out["space_group"] = (
            claim_sg == gt_sg and claim_sg >= 1 and gt_sg >= 1
        )
    except (TypeError, ValueError, KeyError):
        out["space_group"] = False
    return {k: bool(v) for k, v in out.items()}


@app.command()
def main(
    input_path: Path | None = typer.Option(
        None, "--input", "-i",
        help="records.jsonl to repair. Default: latest under "
             "runs/materials/holdout/cot_sft_sae/.",
    ),
    output_path: Path | None = typer.Option(
        None, "--output", "-o",
        help="Default: <input>.repaired.jsonl in the same directory.",
    ),
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
) -> None:
    """Re-parse raw_text in an existing JSONL with the lenient parser."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.ground_truth import is_correct, truth_dict  # noqa: PLC0415
    from fmllm.materials.parse_commit import parse_final_commit  # noqa: PLC0415

    if input_path is None:
        input_path = _latest_records(Path("runs/materials/holdout/cot_sft_sae"))
    if input_path is None or not input_path.exists():
        raise typer.BadParameter(
            "no records.jsonl under runs/materials/holdout/cot_sft_sae/. "
            "Pass --input explicitly."
        )
    if not h5_path.exists():
        raise typer.BadParameter(f"missing {h5_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".repaired.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port: repair parse failures with lenient parser")
    typer.echo(f"    input  : {input_path}")
    typer.echo(f"    h5     : {h5_path}")
    typer.echo(f"    output : {output_path}")
    typer.echo("")

    counters = {
        "total": 0,
        "already_parsed": 0,
        "recovered": 0,
        "still_failed": 0,
        "correct_before": 0,
        "correct_after": 0,
    }

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
            counters["total"] += 1
            if rec.get("is_correct"):
                counters["correct_before"] += 1

            claim = rec.get("claim")
            if claim is not None:
                counters["already_parsed"] += 1
                # Pass-through; preserve existing fields.
                out_f.write(json.dumps(rec) + "\n")
                if rec.get("is_correct"):
                    counters["correct_after"] += 1
                continue

            raw_text = rec.get("raw_text") or ""
            recovered = parse_final_commit(raw_text)
            if recovered is None:
                counters["still_failed"] += 1
                out_f.write(json.dumps(rec) + "\n")
                continue

            counters["recovered"] += 1
            sid = rec.get("specimen_id")
            if isinstance(sid, int) and 0 <= sid < n_h5:
                gt = truth_dict(h5, sid)
            else:
                gt = rec.get("ground_truth") or {}

            correct = is_correct(recovered, gt) if gt else False
            axes = _per_axis_correct(recovered, gt)

            rec["claim"] = recovered
            rec["is_correct"] = bool(correct)
            rec["per_axis_correct"] = axes
            rec["repaired_by_parser"] = True
            rec["repaired_utc"] = datetime.now(UTC).isoformat()
            if correct:
                counters["correct_after"] += 1
            out_f.write(json.dumps(rec) + "\n")

    n = max(counters["total"], 1)
    typer.echo(f"==> Examined {counters['total']} records")
    typer.echo(f"    already parsed   : {counters['already_parsed']}")
    typer.echo(
        f"    recovered        : {counters['recovered']} "
        f"({100.0 * counters['recovered'] / n:.1f}% of total)"
    )
    typer.echo(f"    still failed     : {counters['still_failed']}")
    typer.echo("")
    typer.echo(
        f"    joint correct before : {counters['correct_before']} "
        f"({counters['correct_before'] / n:.4f})"
    )
    typer.echo(
        f"    joint correct after  : {counters['correct_after']} "
        f"({counters['correct_after'] / n:.4f})"
    )
    delta = (counters['correct_after'] - counters['correct_before']) / n
    sign = "+" if delta >= 0 else ""
    typer.echo(f"    delta                : {sign}{delta * 100:.1f} pp")
    typer.echo("")
    typer.echo(f"==> Repaired JSONL: {output_path}")

    summary_path = output_path.with_suffix(".summary.yaml")
    with summary_path.open("w") as f:
        yaml.safe_dump(
            {
                "input_jsonl": str(input_path),
                "output_jsonl": str(output_path),
                "h5_path": str(h5_path),
                "repaired_utc": datetime.now(UTC).isoformat(),
                "counters": counters,
                "joint_before": float(counters["correct_before"] / n),
                "joint_after": float(counters["correct_after"] / n),
                "delta_pp": float(delta * 100),
            },
            f,
            sort_keys=False,
        )
    typer.echo(f"==> Summary: {summary_path}")


if __name__ == "__main__":
    app()
