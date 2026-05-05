"""CLI: probe-head baseline -- direct prediction from FM2 + probes, no LLM.

Phase 16 Stage 4 (the FM-head reference): for each held-out specimen,
forward through FM2, run the trained probe bank, build a final
PhysicalStateClaim directly from the probe outputs (no LLM, no
verifier, no reasoning), and write a one-step trajectory.

The point of this baseline is to ask: how well does the FM's own
downstream head do, with no LLM in the loop? The trained Phase 11
probe bank IS the FM2-side downstream head for our (motif, n_atoms,
T) classification task. Every other baseline runs through this
plus an LLM; this strips the LLM and the verifier.

Output to runs/holdout/probe_head/<run_id>/ so the existing held-out
evaluator picks it up as a new column ``probe_head``.

Decision rules:

  - ``motif`` -> argmax of the motif probe class probabilities
    (the probe's own ``prediction`` already does this).
  - ``n_atoms`` -> round(n_atoms probe regression value) clamped to
    [2, 30] (testbed range).
  - ``temperature`` -> ``phase`` probe disambiguates solid-like vs
    liquid-like; we map to the *centroid* temperature of that phase
    bucket (solid-like = 0.30, liquid-like = 0.80). This is a
    deliberately coarse scalar prediction so the comparison is
    fair: probes don't predict T directly.

Usage:

    bash scripts/run_baseline_probe_head.sh
    SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \\
        bash scripts/run_baseline_probe_head.sh

Depends on:
    typer, torch, h5py, numpy, pyyaml.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.orchestrator.trajectory import (  # noqa: E402
    ActionType,
    LLMAction,
    Step,
    StepType,
    TerminationReason,
    Trajectory,
)
from fmllm.training.probe_bank import ProbeBank  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402
from fmllm.verifier.schema import PhysicalStateClaim  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _latest_fm2(checkpoint_root: Path, train_split: str) -> Path:
    parent = checkpoint_root / "fm2_rdf" / train_split
    cands = sorted(parent.glob("*"), key=lambda p: p.name, reverse=True)
    cands = [c for c in cands if (c / "model.pt").exists()]
    if not cands:
        raise typer.BadParameter(f"no fm2_rdf checkpoint under {parent}")
    return cands[0]


def _claim_from_probes(
    probe_out: dict[str, dict],
    n_min: int = 2,
    n_max: int = 30,
    solid_centroid_t: float = 0.30,
    liquid_centroid_t: float = 0.80,
) -> PhysicalStateClaim:
    """Translate one specimen's probe outputs into a PhysicalStateClaim.

    The probe bank emits a 'prediction' per probe; the rules below
    convert those into the (motif, n_atoms, temperature) tuple.
    """
    n_pred = probe_out.get("n_atoms", {}).get("prediction")
    motif_pred = probe_out.get("motif", {}).get("prediction")
    phase_pred = probe_out.get("phase", {}).get("prediction")
    if isinstance(n_pred, (int, float)):
        n_int = int(round(float(n_pred)))
        n_int = max(n_min, min(n_max, n_int))
    else:
        n_int = -1
    motif_str = (
        str(motif_pred) if motif_pred is not None else "triangular_disk"
    )
    phase_str = str(phase_pred) if phase_pred is not None else "solid-like"
    if "liquid" in phase_str.lower():
        t = float(liquid_centroid_t)
    else:
        t = float(solid_centroid_t)
    return PhysicalStateClaim(
        n_atoms=n_int,
        motif=motif_str,
        temperature=t,
    )


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    checkpoint_root: Path = typer.Option(
        Path("checkpoints"), "--checkpoint-root",
    ),
    train_split: str = typer.Option("train_50k", "--train-split"),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Probe bank directory. Default: latest under checkpoints/probes/.",
    ),
    start: int = typer.Option(0, "--start"),
    count: int = typer.Option(200, "--count"),
    specimen_ids_file: Path | None = typer.Option(
        None, "--specimen-ids-file",
    ),
    out: Path = typer.Option(Path("runs/holdout"), "--out", "-o"),
    batch_size: int = typer.Option(256, "--batch-size"),
    solid_centroid_t: float = typer.Option(0.30, "--solid-centroid-t"),
    liquid_centroid_t: float = typer.Option(0.80, "--liquid-centroid-t"),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(50, "--log-every"),
) -> None:
    """Direct probe-bank prediction baseline."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if probe_bank_dir is None:
        probe_bank_dir = _latest_dir(Path("checkpoints/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter("no probe bank under checkpoints/probes/.")

    if specimen_ids_file is not None:
        with specimen_ids_file.open("r") as f:
            specimen_ids = list(json.load(f))
        run_slug = f"baseline-probe-head-{len(specimen_ids)}-holdout"
    else:
        specimen_ids = list(range(start, start + count))
        run_slug = f"baseline-probe-head-{count}"

    run_id = generate_run_id(run_slug)
    out_dir = out / "probe_head" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> Probe bank  : {probe_bank_dir}")
    typer.echo(f"==> Specimens   : {len(specimen_ids)}")
    typer.echo(
        f"==> T centroids : solid={solid_centroid_t} liquid={liquid_centroid_t}"
    )

    fm2_ckpt = _latest_fm2(checkpoint_root, train_split)
    typer.echo(f"==> FM2 ckpt    : {fm2_ckpt}")
    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    bank = ProbeBank.load(probe_bank_dir, device=device).eval()
    typer.echo(f"    probes loaded: {bank.names()}")

    jsonl_path = out_dir / "trajectories.jsonl"
    counters = {"total": 0, "committed": 0}
    started_run = _now_utc()

    typer.echo(f"==> Generating predictions ({len(specimen_ids)} specimens)")

    with h5py.File(h5_path, "r") as h5, jsonl_path.open("w") as out_f:
        for start_i in range(0, len(specimen_ids), batch_size):
            batch_ids = specimen_ids[start_i : start_i + batch_size]
            rdfs_np = np.stack(
                [np.asarray(h5["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device).float()
            with torch.no_grad():
                hidden = fm2.encode(rdfs)
                cls = hidden[:, 0, :]
            probe_outputs_batch = bank.evaluate(cls)

            for sid, probe_out in zip(batch_ids, probe_outputs_batch, strict=True):
                claim = _claim_from_probes(
                    probe_out,
                    solid_centroid_t=solid_centroid_t,
                    liquid_centroid_t=liquid_centroid_t,
                )
                t_now = _now_utc()
                action = LLMAction(
                    action_type=ActionType.COMMIT,
                    claim=claim,
                    error=None,
                    raw_text="(probe-head direct prediction)",
                )
                step = Step(
                    step_index=0,
                    step_type=StepType.FINAL,
                    timestamp_utc=t_now,
                    llm_action=action,
                    claim=claim,
                )
                traj = Trajectory(
                    run_id=run_id,
                    query="probe_head baseline (FM2 -> probes -> direct claim)",
                    specimen_id=int(sid),
                    started_utc=t_now,
                    finished_utc=t_now,
                    termination=TerminationReason.COMMITTED,
                    final_claim=claim,
                    final_verdict=None,
                    steps=[step],
                    metadata={
                        "baseline": "probe_head",
                        "probe_bank_dir": str(probe_bank_dir),
                        "fm2_checkpoint": str(fm2_ckpt),
                        "probe_outputs": {
                            name: {
                                "prediction": probe_out[name].get("prediction"),
                                "confidence": float(
                                    probe_out[name].get("confidence", 0.0)
                                ),
                            }
                            for name in bank.names()
                        },
                    },
                )
                counters["total"] += 1
                counters["committed"] += 1
                out_f.write(traj.model_dump_json() + "\n")

            if (counters["total"] // batch_size) % max(1, log_every // batch_size) == 0:
                typer.echo(
                    f"    processed {counters['total']}/{len(specimen_ids)}"
                )

    typer.echo(f"==> JSONL: {jsonl_path}")
    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "baseline": "probe_head",
                "counters": counters,
                "started_utc": started_run,
                "finished_utc": _now_utc(),
            },
            f,
        )
    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.run_baseline_probe_head",
        inputs={
            "h5_path": str(h5_path),
            "probe_bank_dir": str(probe_bank_dir),
            "fm2_checkpoint": str(fm2_ckpt),
            "n_specimens": len(specimen_ids),
            "specimen_ids_file": (
                str(specimen_ids_file) if specimen_ids_file is not None else None
            ),
        },
        config={
            "run_id": run_id,
            "solid_centroid_t": solid_centroid_t,
            "liquid_centroid_t": liquid_centroid_t,
        },
        extra={"counters": counters},
    )


if __name__ == "__main__":
    app()
