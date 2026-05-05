"""CLI: emit a JSONL of (probe outputs, synthetic CoT, ground truth)
SFT records.

Phase 11 Stage 1. For every training specimen:

  1. Read the RDF from the dataset HDF5.
  2. Forward through the frozen FM2 backbone (encode).
  3. Run every probe in the bank.
  4. Read ground truth (N, motif, T) from HDF5.
  5. Generate the templated CoT.
  6. Build a (system, user, assistant) chat record and serialize to JSONL.

Output is exactly the shape Phase 6's
:func:`fmllm.training.sft_trainer.train_sft` consumes, so Stage 2
training can call it directly.

Usage:

    bash scripts/build_cot_dataset.sh
    uv run python scripts/build_cot_dataset.py --n-specimens 10000

Depends on:
    typer, torch, h5py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.training.probe_bank import ProbeBank  # noqa: E402
from fmllm.training.synthetic_cot import build_sft_record  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _latest_completed(parent: Path) -> Path | None:
    cands = sorted(
        parent.glob("*"), key=lambda p: p.name, reverse=True,
    )
    return next((c for c in cands if (c / "manifest.yaml").exists()), None)


def _latest_fm2_ckpt(checkpoint_root: Path, train_split: str) -> Path:
    parent = checkpoint_root / "fm2_rdf" / train_split
    cands = sorted(parent.glob("*"), key=lambda p: p.name, reverse=True)
    cands = [c for c in cands if (c / "model.pt").exists()]
    if not cands:
        raise typer.BadParameter(
            f"no fm2_rdf checkpoint under {parent}"
        )
    return cands[0]


def _load_specimen_ids(splits_path: Path, split_name: str) -> list[int]:
    with splits_path.open("r") as f:
        splits = yaml.safe_load(f)
    if split_name == "train":
        return [int(x) for x in splits.get("train", [])]
    sub = splits.get("train_subsets") or {}
    if split_name in sub:
        return [int(x) for x in sub[split_name]]
    raise typer.BadParameter(f"unknown split {split_name!r}")


def _truth_dict(h5: h5py.File, sid: int) -> dict[str, object]:
    motif_id = int(np.asarray(h5["motif_ids"][sid]))
    motif_names = [
        s.decode() if isinstance(s, bytes) else str(s)
        for s in (h5.attrs.get("motif_names") or [])
    ]
    motif = (
        motif_names[motif_id]
        if 0 <= motif_id < len(motif_names) else str(motif_id)
    )
    return {
        "n": int(np.asarray(h5["atom_counts"][sid])),
        "t": float(np.asarray(h5["temperatures"][sid])),
        "motif": motif,
    }


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    splits_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/splits.yaml"), "--splits-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    checkpoint_root: Path = typer.Option(
        Path("checkpoints"), "--checkpoint-root",
    ),
    train_split: str = typer.Option("train_50k", "--train-split"),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Path to a probe bank directory (containing manifest.yaml). "
             "Default: latest under checkpoints/probes/.",
    ),
    n_specimens: int = typer.Option(
        10000, "--n-specimens",
        help="Number of training specimens to emit records for.",
    ),
    out: Path = typer.Option(
        Path("runs/cot_datasets"), "--out", "-o",
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Build the synthetic SFT dataset for Phase 11 Stage 2."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if probe_bank_dir is None:
        probe_bank_dir = _latest_completed(Path("checkpoints/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter(
                "no probe bank under checkpoints/probes/. Run "
                "scripts/train_probe_bank.sh first."
            )

    fm2_ckpt = _latest_fm2_ckpt(checkpoint_root, train_split)

    run_id = generate_run_id(f"cot-dataset-{n_specimens}")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id        : {run_id}")
    typer.echo(f"==> Output        : {out_dir}")
    typer.echo(f"==> FM2 checkpoint: {fm2_ckpt}")
    typer.echo(f"==> Probe bank    : {probe_bank_dir}")

    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    bank = ProbeBank.load(probe_bank_dir, device=device).eval()
    typer.echo(f"    probes loaded: {bank.names()}")

    pool = _load_specimen_ids(splits_path, train_split)
    if not pool:
        raise typer.BadParameter(f"empty split {train_split!r}")
    pool = pool[: max(n_specimens, 1)]
    typer.echo(f"==> Specimens     : {len(pool)}")

    jsonl_path = out_dir / "records.jsonl"
    n_written = 0
    n_consistent = 0
    with h5py.File(h5_path, "r") as h5, jsonl_path.open("w") as out_f:
        for start in range(0, len(pool), batch_size):
            batch_ids = pool[start : start + batch_size]
            rdfs_np = np.stack(
                [np.asarray(h5["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device).float()
            with torch.no_grad():
                hidden = fm2.encode(rdfs)
                cls = hidden[:, 0, :]
            probe_outputs_batch = bank.evaluate(cls)
            for sid, probe_out in zip(batch_ids, probe_outputs_batch, strict=True):
                truth = _truth_dict(h5, sid)
                record = build_sft_record(
                    probe_outputs=probe_out,
                    ground_truth=truth,
                    specimen_id=int(sid),
                )
                if record["cot_consistent"]:
                    n_consistent += 1
                out_f.write(json.dumps(record) + "\n")
                n_written += 1
            if (start // batch_size) % 10 == 0:
                typer.echo(
                    f"    wrote {n_written}/{len(pool)} records "
                    f"(consistent={n_consistent})"
                )

    typer.echo(f"==> JSONL written : {jsonl_path} ({n_written} records)")
    typer.echo(
        f"    coord-consistent : {n_consistent} "
        f"({100.0 * n_consistent / max(n_written, 1):.1f}%)"
    )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.build_cot_dataset",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "fm2_checkpoint": str(fm2_ckpt),
            "probe_bank_dir": str(probe_bank_dir),
            "train_split": train_split,
            "n_specimens": len(pool),
        },
        config={
            "run_id": run_id,
            "batch_size": batch_size,
        },
        extra={
            "n_records": n_written,
            "n_coord_consistent": n_consistent,
            "jsonl_path": str(jsonl_path),
        },
    )


if __name__ == "__main__":
    app()
