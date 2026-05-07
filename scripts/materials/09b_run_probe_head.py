"""Stage 9b: probe-head baseline (no LLM).

Mirrors ``scripts/materials/09_run_singleshot.py`` but with the
LLM stack stripped out. For each held-out specimen:

  1. Forward CHGNet (live).
  2. Run the materials probe bank.
  3. Build the joint claim directly from probe outputs:

         formation_energy = formation_energy probe prediction
         e_above_hull     = e_above_hull probe prediction
         is_stable        = (e_above_hull_pred <= 0.025)
         band_gap_class   = is_metal probe + band_gap probe disambiguation
         space_group      = space_group probe argmax

  4. Score via ``fmllm.materials.ground_truth.is_correct`` -- same
     joint correctness criterion as Stage 9.

This is the "no-LLM floor": what we'd report if we just used
CHGNet's downstream supervised heads. If Stage 9's cot_sft_sae
beats this, the LLM is adding value over the FM. If it doesn't,
the LLM is at best laundering CHGNet's predictions.

Output:

    runs/materials/holdout/probe_head/<run_id>/records.jsonl
    runs/materials/holdout/probe_head/<run_id>/summary.yaml
    runs/materials/holdout/probe_head/<run_id>/manifest.yaml

Usage:

    bash scripts/materials/09b_run_probe_head.sh

Depends on:
    typer, h5py, numpy, torch, pyyaml, chgnet, pymatgen.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _generate_run_id(slug: str) -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{slug}"


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _claim_from_probes(probe_out: dict) -> dict:
    """Synthesize the joint materials claim from probe outputs.

    Uses the is_metal probe + band_gap probe jointly for
    band_gap_class to maximize the probe-only accuracy ceiling.
    """
    e_form = float(probe_out["formation_energy"]["prediction"])
    e_hull = float(probe_out["e_above_hull"]["prediction"])
    is_stable = bool(e_hull <= 0.025)

    bg = float(probe_out["band_gap"]["prediction"])
    is_metal_pred = str(probe_out["is_metal"]["prediction"]).lower()
    if is_metal_pred == "metal":
        bg_class = "metal"
    elif bg > 3.0:
        bg_class = "wide"
    else:
        bg_class = "narrow"

    sg_pred = str(probe_out.get("space_group", {}).get("prediction", ""))
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


def _per_axis_correct(claim: dict, gt: dict) -> dict:
    """Per-axis breakdown of correctness for diagnostic purposes."""
    return {
        "formation_energy": (
            abs(float(claim["formation_energy"]) - gt["formation_energy"]) <= 0.05
        ),
        "e_above_hull": (
            abs(float(claim["e_above_hull"]) - gt["e_above_hull"]) <= 0.025
        ),
        "is_stable": bool(claim["is_stable"]) == bool(gt["is_stable"]),
        "band_gap_class": (
            str(claim["band_gap_class"]).lower() == gt["band_gap_class"].lower()
        ),
        "space_group": int(claim["space_group"]) == gt["space_group"],
    }


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    holdout_ids_path: Path = typer.Option(
        Path("data/materials_project_v1/holdout_lock/ids.json"),
        "--holdout-ids-path",
    ),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Default: latest under checkpoints/materials/probes/.",
    ),
    chgnet_model_name: str = typer.Option("0.3.0", "--chgnet-model-name"),
    max_atoms: int = typer.Option(80, "--max-atoms"),
    out: Path = typer.Option(
        Path("runs/materials/holdout"), "--out", "-o",
    ),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(20, "--log-every"),
) -> None:
    """Probe-head baseline: CHGNet supervised heads alone, no LLM."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.chgnet_wrap import (  # noqa: PLC0415
        CHGNetWrap, structure_from_arrays,
    )
    from fmllm.materials.ground_truth import is_correct, truth_dict  # noqa: PLC0415
    from fmllm.training.probe_bank import ProbeBank  # noqa: PLC0415

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if probe_bank_dir is None:
        probe_bank_dir = _latest_dir(Path("checkpoints/materials/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter(
                "no probe bank under checkpoints/materials/probes/."
            )

    if not holdout_ids_path.exists():
        raise typer.BadParameter(f"missing {holdout_ids_path}")
    with holdout_ids_path.open("r") as f:
        holdout_ids = [int(s) for s in json.load(f)]

    run_id = _generate_run_id(f"mat-probe-head-{len(holdout_ids)}-holdout")
    out_dir = out / "probe_head" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 9b: probe-head baseline (no LLM)")
    typer.echo(f"    Run id          : {run_id}")
    typer.echo(f"    Output          : {out_dir}")
    typer.echo(f"    Probe bank      : {probe_bank_dir}")
    typer.echo(f"    Holdout ids     : {len(holdout_ids)}")
    typer.echo("")

    typer.echo("==> Loading CHGNet...")
    wrap = CHGNetWrap.load(device=device, model_name=chgnet_model_name)
    bank = ProbeBank.load(probe_bank_dir, device=device).eval()
    typer.echo(f"    probes loaded   : {bank.names()}")

    jsonl_path = out_dir / "records.jsonl"
    counters = {
        "total": 0,
        "correct": 0,
        "skipped_chgnet_error": 0,
    }
    per_axis_correct = {
        "formation_energy": 0,
        "e_above_hull": 0,
        "is_stable": 0,
        "band_gap_class": 0,
        "space_group": 0,
    }
    started_run = _now_utc()

    typer.echo(f"==> Starting baseline ({len(holdout_ids)} specimens)")
    with h5py.File(h5_path, "r") as h5, jsonl_path.open("w") as out_f:
        element_names_attr = h5.attrs.get("element_names")
        element_names = (
            [s.decode() if isinstance(s, bytes) else str(s)
             for s in element_names_attr]
            if element_names_attr is not None else []
        )

        for sid in holdout_ids:
            n_atoms = int(np.asarray(h5["nsites"][sid]))
            if n_atoms > max_atoms or n_atoms < 1:
                counters["skipped_chgnet_error"] += 1
                continue
            species_ids = np.asarray(h5["n_atoms_padded"][sid])[:n_atoms]
            positions = np.asarray(h5["positions_padded"][sid])[:n_atoms]
            lattice = np.asarray(h5["lattice"][sid])
            try:
                structure = structure_from_arrays(
                    species_ids=species_ids,
                    positions=positions,
                    lattice=lattice,
                    element_names=element_names,
                )
                _, pooled = wrap.encode(structure)
            except Exception as exc:
                counters["skipped_chgnet_error"] += 1
                if counters["skipped_chgnet_error"] <= 5:
                    typer.echo(f"    skip sid={sid}: {exc!r}")
                continue

            x = pooled.detach().to(device).float().reshape(1, -1)
            probe_outputs_batch = bank.evaluate(x)
            probe_out = probe_outputs_batch[0]

            claim = _claim_from_probes(probe_out)
            truth = truth_dict(h5, int(sid))
            correct = is_correct(claim, truth)
            axis_correct = _per_axis_correct(claim, truth)

            for axis, ok in axis_correct.items():
                if ok:
                    per_axis_correct[axis] += 1

            record = {
                "specimen_id": int(sid),
                "claim": claim,
                "is_correct": bool(correct),
                "per_axis_correct": {k: bool(v) for k, v in axis_correct.items()},
                "ground_truth": {
                    k: (
                        bool(v) if isinstance(v, np.bool_)
                        else float(v) if isinstance(v, (np.floating, float))
                        else int(v) if isinstance(v, (np.integer, int))
                        else str(v)
                    )
                    for k, v in truth.items()
                },
                "probe_outputs": {
                    name: {
                        "prediction": probe_out[name].get("prediction"),
                        "confidence": float(
                            probe_out[name].get("confidence", 0.0)
                        ),
                    }
                    for name in bank.names()
                },
            }
            counters["total"] += 1
            if correct:
                counters["correct"] += 1
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (
                counters["total"] == 1
                or counters["total"] % log_every == 0
                or counters["total"] == len(holdout_ids)
            ):
                typer.echo(
                    f"    {counters['total']:>4}/{len(holdout_ids)} "
                    f"sid={int(sid):<8} "
                    f"correct={counters['correct']} "
                    f"chgnet_skip={counters['skipped_chgnet_error']}"
                )

    accuracy = counters["correct"] / max(counters["total"], 1)
    typer.echo(f"==> JSONL: {jsonl_path}")
    typer.echo(
        f"    joint accuracy: {counters['correct']}/{counters['total']} "
        f"= {accuracy:.4f}"
    )
    typer.echo("    per-axis accuracy:")
    for axis, n_ok in per_axis_correct.items():
        typer.echo(
            f"      {axis:<20s}: {n_ok}/{counters['total']} "
            f"= {n_ok / max(counters['total'], 1):.4f}"
        )

    summary = {
        "baseline": "probe_head",
        "domain": "materials",
        "counters": counters,
        "accuracy": float(accuracy),
        "per_axis_accuracy": {
            k: float(v / max(counters["total"], 1))
            for k, v in per_axis_correct.items()
        },
        "started_utc": started_run,
        "finished_utc": _now_utc(),
    }
    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    with (out_dir / "manifest.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "completed_utc": datetime.now(UTC).isoformat(),
                "h5_path": str(h5_path),
                "holdout_ids_path": str(holdout_ids_path),
                "probe_bank_dir": str(probe_bank_dir),
                "chgnet_model_name": chgnet_model_name,
                "max_atoms": max_atoms,
                "n_holdout": len(holdout_ids),
                "counters": counters,
                "accuracy": float(accuracy),
                "per_axis_accuracy": summary["per_axis_accuracy"],
            },
            f,
            sort_keys=False,
        )


if __name__ == "__main__":
    app()
