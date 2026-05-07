"""Stage 9c: hybrid LLM+probe scoring (no re-inference needed).

For each specimen in an existing cot_sft_sae records.jsonl, build
a HYBRID claim:

  formation_energy : LLM's claim   (regression: LLM helps)
  e_above_hull     : LLM's claim   (regression: LLM helps)
  is_stable        : probe rule    (e_above_hull_pred <= 0.025)
  band_gap_class   : probe rule    (is_metal + band_gap binning)
  space_group      : probe argmax  (top-20 head)

The LLM's classification axes (is_stable / band_gap_class /
space_group) are *replaced* with the probe's rule-based answer.
This keeps the LLM only where it adds value (regression refinement
+4-6 pp) and uses the probes where they outperform (~6-12 pp).

If the LLM failed to parse for a specimen, the hybrid uses
probe_head answers for ALL axes (full probe_head fallback).

This is the "what's the LLM actually contributing?" diagnostic.
If the hybrid joint accuracy clearly beats both probe_head AND
cot_sft_sae, the recipe IS adding value -- it's just that the
single-shot SFT'd LLM degrades classification axes that the
probes handle well, and a hybrid evaluator separates the two.

Output:

    runs/materials/holdout/hybrid/<run_id>/records.jsonl
    runs/materials/holdout/hybrid/<run_id>/summary.yaml

Usage:

    bash scripts/materials/09c_score_hybrid.sh
    bash scripts/materials/09c_score_hybrid.sh --input <cot_sft_sae.jsonl>

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


def _generate_run_id(slug: str) -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{slug}"


def _latest_records(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*/records.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


def _probe_claim(probe_outputs: dict) -> dict:
    """Build a probe_head-style claim from stored probe outputs.

    Mirrors 09b._claim_from_probes exactly so the hybrid baseline
    is directly comparable.
    """
    e_form = float(probe_outputs["formation_energy"]["prediction"])
    e_hull = float(probe_outputs["e_above_hull"]["prediction"])
    is_stable = bool(e_hull <= 0.025)

    bg = float(probe_outputs["band_gap"]["prediction"])
    is_metal_pred = str(probe_outputs["is_metal"]["prediction"]).lower()
    if is_metal_pred == "metal":
        bg_class = "metal"
    elif bg > 3.0:
        bg_class = "wide"
    else:
        bg_class = "narrow"

    sg_pred = str(probe_outputs.get("space_group", {}).get("prediction", ""))
    if sg_pred.startswith("sg"):
        try:
            sg_int = int(sg_pred[2:])
        except ValueError:
            sg_int = -1
    else:
        try:
            sg_int = int(sg_pred)
        except (ValueError, TypeError):
            sg_int = -1

    return {
        "formation_energy": e_form,
        "e_above_hull": e_hull,
        "is_stable": is_stable,
        "band_gap_class": bg_class,
        "space_group": sg_int,
    }


def _hybrid_claim(llm_claim: dict | None, probe_outputs: dict) -> dict:
    """LLM's regression + probe's classification.

    The is_stable axis stays bound to the PROBE's e_above_hull
    threshold. Using the LLM's e_above_hull instead flips ~10% of
    specimens near the 0.025 boundary even though the LLM's value
    is more often within +/-0.025 of truth -- the threshold-cross
    error rate is decoupled from the regression MAE. Empirically
    on the materials holdout, probe-thresholded is_stable beat
    LLM-thresholded by ~9.5 pp.
    """
    probe = _probe_claim(probe_outputs)
    if llm_claim is None:
        return probe
    out = dict(probe)  # start from probe baseline
    # Override regression axes with LLM where present and numeric.
    try:
        out["formation_energy"] = float(llm_claim["formation_energy"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        out["e_above_hull"] = float(llm_claim["e_above_hull"])
    except (KeyError, TypeError, ValueError):
        pass
    # is_stable follows the PROBE's e_above_hull threshold (already
    # set by _probe_claim above) -- do NOT override with LLM-derived
    # value. This is the classification-side decision.
    return out


def _per_axis_correct(claim: dict, gt: dict) -> dict:
    """Same per-axis criterion as 09b / repair / compare."""
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
        help="cot_sft_sae records.jsonl. Default: latest under "
             "runs/materials/holdout/cot_sft_sae/. Use the .repaired.jsonl "
             "version if available -- it has more parsed claims.",
    ),
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    out: Path = typer.Option(
        Path("runs/materials/holdout"), "--out", "-o",
    ),
) -> None:
    """Score the hybrid LLM+probe claim and report joint+per-axis."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.ground_truth import is_correct, truth_dict  # noqa: PLC0415

    if input_path is None:
        # Prefer .repaired.jsonl if both exist for the latest run.
        cot_dir = Path("runs/materials/holdout/cot_sft_sae")
        latest = (
            sorted(cot_dir.glob("*"), key=lambda p: p.stat().st_mtime,
                   reverse=True)[:1]
        )
        if not latest:
            raise typer.BadParameter(
                "no cot_sft_sae run under runs/materials/holdout/cot_sft_sae/"
            )
        run_dir = latest[0]
        repaired = run_dir / "records.repaired.jsonl"
        plain = run_dir / "records.jsonl"
        input_path = repaired if repaired.exists() else plain
    if not input_path.exists():
        raise typer.BadParameter(f"missing {input_path}")
    if not h5_path.exists():
        raise typer.BadParameter(f"missing {h5_path}")

    run_id = _generate_run_id("mat-hybrid-200-holdout")
    out_dir = out / "hybrid" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 9c: hybrid LLM+probe scoring")
    typer.echo(f"    input  : {input_path}")
    typer.echo(f"    h5     : {h5_path}")
    typer.echo(f"    output : {out_dir}")
    typer.echo("")

    counters = {
        "total": 0,
        "with_llm_claim": 0,
        "without_llm_claim": 0,
        "correct": 0,
    }
    per_axis_correct: dict[str, int] = {
        ax: 0 for ax in (
            "formation_energy", "e_above_hull", "is_stable",
            "band_gap_class", "space_group",
        )
    }

    jsonl_path = out_dir / "records.jsonl"
    n_h5 = 0
    with h5py.File(h5_path, "r") as h5, \
            input_path.open("r") as in_f, jsonl_path.open("w") as out_f:
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
                continue
            probe_outputs = rec.get("probe_outputs") or {}
            llm_claim = rec.get("claim")

            hybrid = _hybrid_claim(llm_claim, probe_outputs)
            gt = truth_dict(h5, sid)
            correct = is_correct(hybrid, gt)
            axes = _per_axis_correct(hybrid, gt)

            counters["total"] += 1
            if llm_claim is not None:
                counters["with_llm_claim"] += 1
            else:
                counters["without_llm_claim"] += 1
            if correct:
                counters["correct"] += 1
            for axis, ok in axes.items():
                if ok:
                    per_axis_correct[axis] += 1

            out_f.write(json.dumps({
                "specimen_id": sid,
                "claim": hybrid,
                "is_correct": bool(correct),
                "per_axis_correct": axes,
                "ground_truth": {
                    k: (
                        bool(v) if isinstance(v, np.bool_)
                        else float(v) if isinstance(v, (np.floating, float))
                        else int(v) if isinstance(v, (np.integer, int))
                        else str(v)
                    )
                    for k, v in gt.items()
                },
                "probe_outputs": probe_outputs,
                "source": (
                    "hybrid_llm_regression_+_probe_classification"
                    if llm_claim is not None else "probe_only_fallback"
                ),
            }) + "\n")

    n = max(counters["total"], 1)
    accuracy = counters["correct"] / n
    typer.echo(f"==> Examined {counters['total']} records")
    typer.echo(f"    with LLM claim   : {counters['with_llm_claim']}")
    typer.echo(f"    without LLM claim: {counters['without_llm_claim']} "
               f"(probe_head fallback used)")
    typer.echo("")
    typer.echo(f"    joint accuracy : {counters['correct']}/{n} = "
               f"{accuracy:.4f}")
    typer.echo("    per-axis accuracy:")
    for axis in per_axis_correct:
        n_ok = per_axis_correct[axis]
        typer.echo(f"      {axis:<20s}: {n_ok:>4d}/{n:<4d} = {n_ok / n:.4f}")

    typer.echo("")
    typer.echo(f"==> JSONL: {jsonl_path}")

    summary = {
        "baseline": "hybrid",
        "domain": "materials",
        "input_jsonl": str(input_path),
        "completed_utc": datetime.now(UTC).isoformat(),
        "counters": counters,
        "accuracy": float(accuracy),
        "per_axis_accuracy": {
            k: float(per_axis_correct[k] / n) for k in per_axis_correct
        },
    }
    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    with (out_dir / "manifest.yaml").open("w") as f:
        yaml.safe_dump({**summary, "run_id": run_id}, f, sort_keys=False)


if __name__ == "__main__":
    app()
