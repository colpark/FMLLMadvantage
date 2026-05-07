"""Compare materials baselines side-by-side.

Reads the JSONL outputs from any combination of Stage 9 (cot_sft_sae)
and Stage 9b (probe_head), aligns on specimen_id, and prints:

  * Joint accuracy per baseline.
  * Per-axis accuracy per baseline (where available).
  * Delta vs probe_head (the LLM contribution).
  * Per-specimen disagreement breakdown.

By default picks the latest run of each baseline under
``runs/materials/holdout/<baseline>/``.

Usage:

    bash scripts/materials/compare_baselines.sh
    uv run python scripts/materials/compare_baselines.py \\
        --probe-head-jsonl runs/materials/holdout/probe_head/<run_id>/records.jsonl \\
        --cot-sft-sae-jsonl runs/materials/holdout/cot_sft_sae/<run_id>/records.jsonl

Depends on:
    Stdlib + typer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer


app = typer.Typer(add_completion=False, no_args_is_help=False)


_AXES = ["formation_energy", "e_above_hull", "is_stable",
         "band_gap_class", "space_group"]


def _latest_jsonl(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*/records.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


def _per_axis_correct_from_record(rec: dict) -> dict[str, bool]:
    """Return per-axis correctness for a record.

    probe_head emits per_axis_correct directly; cot_sft_sae records
    only carry is_correct, so we recompute axis-wise from claim+gt.
    """
    if "per_axis_correct" in rec:
        return {k: bool(v) for k, v in rec["per_axis_correct"].items()}
    claim = rec.get("claim") or {}
    gt = rec.get("ground_truth") or {}
    if not claim:
        return {axis: False for axis in _AXES}
    out: dict[str, bool] = {}
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
    return out


def _read_jsonl(path: Path) -> dict[int, dict]:
    """Index records by specimen_id."""
    out: dict[int, dict] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("specimen_id")
            if isinstance(sid, int):
                out[sid] = rec
    return out


def _summarize(records: dict[int, dict]) -> dict:
    """Return overall + per-axis accuracy."""
    n = len(records)
    if n == 0:
        return {
            "n": 0, "joint": 0.0, "joint_correct": 0,
            "per_axis": {axis: 0.0 for axis in _AXES},
            "per_axis_correct": {axis: 0 for axis in _AXES},
            "parse_failure": 0,
        }
    joint_correct = sum(1 for r in records.values() if r.get("is_correct"))
    parse_failure = sum(1 for r in records.values() if r.get("claim") is None)
    per_axis_correct: dict[str, int] = {axis: 0 for axis in _AXES}
    for rec in records.values():
        axes = _per_axis_correct_from_record(rec)
        for axis, ok in axes.items():
            if ok:
                per_axis_correct[axis] += 1
    return {
        "n": n,
        "joint": joint_correct / n,
        "joint_correct": joint_correct,
        "per_axis": {axis: per_axis_correct[axis] / n for axis in _AXES},
        "per_axis_correct": per_axis_correct,
        "parse_failure": parse_failure,
    }


def _print_table(name: str, summary: dict) -> None:
    typer.echo(f"\n=== {name} ===")
    typer.echo(f"  n={summary['n']}  joint accuracy: "
               f"{summary['joint_correct']}/{summary['n']} "
               f"= {summary['joint']:.4f}")
    if summary["parse_failure"] > 0:
        typer.echo(f"  parse failures : {summary['parse_failure']}")
    typer.echo("  per-axis accuracy:")
    for axis in _AXES:
        n_ok = summary["per_axis_correct"][axis]
        frac = summary["per_axis"][axis]
        typer.echo(f"    {axis:<20s}: {n_ok:>4d}/{summary['n']:<4d} "
                   f"= {frac:.4f}")


def _print_delta(
    name_a: str, sum_a: dict, name_b: str, sum_b: dict,
) -> None:
    """Print per-axis delta (b - a). Positive means b > a."""
    typer.echo(f"\n=== Delta: {name_b} - {name_a} ===")
    j_delta = sum_b["joint"] - sum_a["joint"]
    sign = "+" if j_delta >= 0 else ""
    typer.echo(f"  joint              : {sign}{j_delta * 100:.1f} pp "
               f"({sum_a['joint']:.4f} -> {sum_b['joint']:.4f})")
    typer.echo("  per-axis           :")
    for axis in _AXES:
        d = sum_b["per_axis"][axis] - sum_a["per_axis"][axis]
        sign = "+" if d >= 0 else ""
        typer.echo(f"    {axis:<20s}: {sign}{d * 100:.1f} pp "
                   f"({sum_a['per_axis'][axis]:.4f} -> "
                   f"{sum_b['per_axis'][axis]:.4f})")


def _print_disagreement(
    name_a: str, recs_a: dict[int, dict],
    name_b: str, recs_b: dict[int, dict],
) -> None:
    """For specimens evaluated by both, count joint-correctness disagreements."""
    common = sorted(set(recs_a.keys()) & set(recs_b.keys()))
    if not common:
        return
    a_only = sum(
        1 for sid in common
        if recs_a[sid].get("is_correct") and not recs_b[sid].get("is_correct")
    )
    b_only = sum(
        1 for sid in common
        if recs_b[sid].get("is_correct") and not recs_a[sid].get("is_correct")
    )
    both = sum(
        1 for sid in common
        if recs_a[sid].get("is_correct") and recs_b[sid].get("is_correct")
    )
    neither = sum(
        1 for sid in common
        if not recs_a[sid].get("is_correct")
        and not recs_b[sid].get("is_correct")
    )
    typer.echo(f"\n=== Joint-correctness overlap ({len(common)} common) ===")
    typer.echo(f"  both correct          : {both}")
    typer.echo(f"  {name_a} only         : {a_only}")
    typer.echo(f"  {name_b} only         : {b_only}")
    typer.echo(f"  neither correct       : {neither}")
    if (a_only + b_only) > 0:
        net_lift = b_only - a_only
        sign = "+" if net_lift >= 0 else ""
        typer.echo(f"  net lift {name_b} - {name_a}: {sign}{net_lift} specimens")


@app.command()
def main(
    probe_head_jsonl: Path | None = typer.Option(
        None, "--probe-head-jsonl",
        help="Path to probe_head/<run>/records.jsonl. "
             "Default: latest.",
    ),
    cot_sft_sae_jsonl: Path | None = typer.Option(
        None, "--cot-sft-sae-jsonl",
        help="Path to cot_sft_sae/<run>/records.jsonl. "
             "Default: latest.",
    ),
    out_root: Path = typer.Option(
        Path("runs/materials/holdout"), "--out-root",
    ),
) -> None:
    """Print a side-by-side comparison of materials baselines."""
    if probe_head_jsonl is None:
        probe_head_jsonl = _latest_jsonl(out_root / "probe_head")
    if cot_sft_sae_jsonl is None:
        cot_sft_sae_jsonl = _latest_jsonl(out_root / "cot_sft_sae")

    if probe_head_jsonl is None and cot_sft_sae_jsonl is None:
        typer.echo("ERROR: no records.jsonl found under either "
                   "runs/materials/holdout/probe_head/ or "
                   "runs/materials/holdout/cot_sft_sae/.", err=True)
        sys.exit(1)

    if probe_head_jsonl is not None:
        typer.echo(f"probe_head      : {probe_head_jsonl}")
    if cot_sft_sae_jsonl is not None:
        typer.echo(f"cot_sft_sae     : {cot_sft_sae_jsonl}")

    recs_probe = (
        _read_jsonl(probe_head_jsonl) if probe_head_jsonl is not None else {}
    )
    recs_llm = (
        _read_jsonl(cot_sft_sae_jsonl) if cot_sft_sae_jsonl is not None else {}
    )

    sum_probe = _summarize(recs_probe) if recs_probe else None
    sum_llm = _summarize(recs_llm) if recs_llm else None

    if sum_probe is not None:
        _print_table("probe_head (CHGNet heads only)", sum_probe)
    if sum_llm is not None:
        _print_table("cot_sft_sae (probes + SAE -> Qwen+LoRA)", sum_llm)
    if sum_probe is not None and sum_llm is not None:
        _print_delta("probe_head", sum_probe, "cot_sft_sae", sum_llm)
        _print_disagreement(
            "probe_head", recs_probe,
            "cot_sft_sae", recs_llm,
        )


if __name__ == "__main__":
    app()
