"""CLI: emit a JSONL of (probe + SAE-feature, synthetic CoT, ground truth) records.

Phase 16 Stage 1. Same as scripts/build_cot_dataset.py, but each
record additionally carries the top-K active labelled SAE features
for the specimen. The synthetic CoT references *both* probes (Step 1)
and SAE features (Step 1b), and the user message at training time
includes both PROBES and SAE_FEATURES payloads.

The intent is to give the SFT trainer richer evidence per specimen,
testing whether the LLM trained on (probe + SAE) CoT outperforms

  (a) the LLM trained on probes-only CoT (Phase 11), and
  (b) the FM's own downstream head (probe ensemble, Phase 16
      ``probe_head`` baseline).

No verifier is involved at any point in this Phase 16 line of work;
the LLM is evaluated single-shot with the trained adapter and the
typed prompt.

Output:

    runs/cot_datasets_sae/<run_id>/records.jsonl
    runs/cot_datasets_sae/<run_id>/manifest.yaml

Usage:

    bash scripts/build_cot_dataset_with_sae.sh
    uv run python scripts/build_cot_dataset_with_sae.py --n-specimens 10000

Depends on:
    typer, torch, h5py, pyyaml, numpy.
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
from fmllm.representation.sae import TopKSAE  # noqa: E402
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


def _latest_dir_with(parent: Path, sub: str) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob(f"*/{sub}"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0].parent if cands else None


def _latest_fm2_ckpt(checkpoint_root: Path, train_split: str) -> Path:
    parent = checkpoint_root / "fm2_rdf" / train_split
    cands = sorted(parent.glob("*"), key=lambda p: p.name, reverse=True)
    cands = [c for c in cands if (c / "model.pt").exists()]
    if not cands:
        raise typer.BadParameter(f"no fm2_rdf checkpoint under {parent}")
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
    motif_names: list[str] = []
    if "motif_names" in h5.attrs:
        motif_names = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in h5.attrs["motif_names"]
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


def _load_sae(sae_path: Path, device: str) -> tuple[TopKSAE, torch.Tensor, torch.Tensor]:
    payload = torch.load(sae_path, map_location=device, weights_only=False)
    sae = TopKSAE(
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        k=int(payload["k"]),
    ).to(device)
    sae.load_state_dict(payload["state_dict"], strict=True)
    sae.eval()
    cls_mean = torch.from_numpy(
        np.asarray(payload["cls_mean"], dtype=np.float32),
    ).to(device).flatten()
    cls_std = torch.from_numpy(
        np.asarray(payload["cls_std"], dtype=np.float32),
    ).to(device).flatten()
    return sae, cls_mean, cls_std


def _load_labels(labels_path: Path) -> dict[int, str]:
    with labels_path.open("r") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def _top_k_sae_features(
    z: torch.Tensor,                # (B, hidden_dim)
    labels: dict[int, str],
    top_k: int,
) -> list[list[tuple[str, float]]]:
    """For each row of ``z``, return the top-k active features as
    a list of (label, activation) tuples sorted by activation desc."""
    out: list[list[tuple[str, float]]] = []
    z_np = z.detach().cpu().numpy()
    for row in z_np:
        nz = np.nonzero(row)[0]
        if nz.size == 0:
            out.append([])
            continue
        nz_acts = row[nz]
        order = np.argsort(nz_acts)[::-1][:top_k]
        top_idx = nz[order]
        top_act = nz_acts[order]
        out.append([
            (labels.get(int(i), f"f{int(i)}"), float(a))
            for i, a in zip(top_idx, top_act, strict=True)
        ])
    return out


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
        help="Path to the probe bank. Default: latest under checkpoints/probes/.",
    ),
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained SAE directory (containing sae.pt). Default: latest "
             "under checkpoints/sae/.",
    ),
    sae_labels_path: Path | None = typer.Option(
        None, "--sae-labels-path",
        help="labels.json for the SAE. Default: latest under runs/sae_labels/.",
    ),
    top_k_features: int = typer.Option(
        8, "--top-k-features",
        help="How many top-active SAE features to surface per specimen.",
    ),
    n_specimens: int = typer.Option(
        10000, "--n-specimens",
        help="Number of training specimens to emit records for.",
    ),
    out: Path = typer.Option(
        Path("runs/cot_datasets_sae"), "--out", "-o",
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Build the SAE-augmented synthetic SFT dataset."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if probe_bank_dir is None:
        probe_bank_dir = _latest_completed(Path("checkpoints/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter(
                "no probe bank under checkpoints/probes/."
            )

    if sae_dir is None:
        sae_dir = _latest_completed(Path("checkpoints/sae"))
        if sae_dir is None:
            raise typer.BadParameter(
                "no SAE under checkpoints/sae/. Run scripts/train_sae.sh first."
            )
    sae_path = sae_dir / "sae.pt"
    if not sae_path.exists():
        raise typer.BadParameter(f"no sae.pt under {sae_dir}")

    if sae_labels_path is None:
        cands = sorted(
            Path("runs/sae_labels").glob("*/labels.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        sae_labels_path = cands[0] if cands else None
    if sae_labels_path is None or not sae_labels_path.exists():
        raise typer.BadParameter(
            "no SAE labels found. Run scripts/label_sae_features.sh first."
        )

    fm2_ckpt = _latest_fm2_ckpt(checkpoint_root, train_split)

    run_id = generate_run_id(f"cot-sae-dataset-{n_specimens}")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id        : {run_id}")
    typer.echo(f"==> Output        : {out_dir}")
    typer.echo(f"==> FM2           : {fm2_ckpt}")
    typer.echo(f"==> Probe bank    : {probe_bank_dir}")
    typer.echo(f"==> SAE           : {sae_path}")
    typer.echo(f"==> Labels        : {sae_labels_path}")
    typer.echo(f"==> Top-K features: {top_k_features}")

    # Models -----------------------------------------------------------------
    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    bank = ProbeBank.load(probe_bank_dir, device=device).eval()
    typer.echo(f"    probes loaded : {bank.names()}")

    sae, cls_mean, cls_std = _load_sae(sae_path, device=device)
    typer.echo(
        f"    SAE config    : in_dim={sae.in_dim} "
        f"hidden_dim={sae.hidden_dim} k={sae.k}"
    )
    labels = _load_labels(sae_labels_path)

    pool = _load_specimen_ids(splits_path, train_split)
    if not pool:
        raise typer.BadParameter(f"empty split {train_split!r}")
    pool = pool[: max(n_specimens, 1)]
    typer.echo(f"==> Specimens     : {len(pool)}")

    # Iterate ----------------------------------------------------------------
    jsonl_path = out_dir / "records.jsonl"
    n_written = 0
    n_consistent = 0
    n_with_sae = 0
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
                cls_norm = (cls - cls_mean) / cls_std.clamp_min(1.0e-6)
                z = sae.encode(cls_norm)              # (B, hidden_dim)
            probe_outputs_batch = bank.evaluate(cls)
            sae_features_batch = _top_k_sae_features(
                z, labels=labels, top_k=top_k_features,
            )
            for sid, probe_out, sae_feat in zip(
                batch_ids, probe_outputs_batch, sae_features_batch, strict=True,
            ):
                truth = _truth_dict(h5, sid)
                record = build_sft_record(
                    probe_outputs=probe_out,
                    ground_truth=truth,
                    specimen_id=int(sid),
                    sae_features=sae_feat,
                )
                if record["cot_consistent"]:
                    n_consistent += 1
                if record["sae_features_count"] > 0:
                    n_with_sae += 1
                out_f.write(json.dumps(record) + "\n")
                n_written += 1
            if (start // batch_size) % 10 == 0:
                typer.echo(
                    f"    wrote {n_written}/{len(pool)} records "
                    f"(consistent={n_consistent} with-SAE={n_with_sae})"
                )

    typer.echo(f"==> JSONL written : {jsonl_path} ({n_written} records)")
    typer.echo(
        f"    coord-consistent : {n_consistent} "
        f"({100.0 * n_consistent / max(n_written, 1):.1f}%)"
    )
    typer.echo(
        f"    with SAE feats   : {n_with_sae} "
        f"({100.0 * n_with_sae / max(n_written, 1):.1f}%)"
    )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.build_cot_dataset_with_sae",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "fm2_checkpoint": str(fm2_ckpt),
            "probe_bank_dir": str(probe_bank_dir),
            "sae_dir": str(sae_dir),
            "sae_labels_path": str(sae_labels_path),
            "train_split": train_split,
            "n_specimens": len(pool),
        },
        config={
            "run_id": run_id,
            "batch_size": batch_size,
            "top_k_features": top_k_features,
        },
        extra={
            "n_records": n_written,
            "n_coord_consistent": n_consistent,
            "n_with_sae_features": n_with_sae,
            "jsonl_path": str(jsonl_path),
        },
    )


if __name__ == "__main__":
    app()
