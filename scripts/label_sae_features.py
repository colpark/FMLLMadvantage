"""CLI: label every SAE feature by correlation with dataset attributes.

Phase 13 Stage 1. For each feature in the trained SAE codebook, find
the specimens that activate it most strongly and describe what those
specimens have in common using motif / atom-count / temperature /
phase. Emits a labels JSON the inference path consumes.

Output:

    runs/sae_labels/<run_id>/labels.json     # feature_idx -> label string
    runs/sae_labels/<run_id>/details.yaml    # full FeatureLabel records
    runs/sae_labels/<run_id>/manifest.yaml

Usage:

    bash scripts/label_sae_features.sh
    uv run python scripts/label_sae_features.py --top-n 50

Depends on:
    typer, torch, numpy, h5py, pyyaml.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.connectors.text_annotations import _phase_for  # noqa: E402
from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.representation.labels import label_feature  # noqa: E402
from fmllm.representation.sae import TopKSAE  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


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


def _load_specimen_ids(splits_path: Path, split_name: str) -> list[int]:
    with splits_path.open("r") as f:
        splits = yaml.safe_load(f)
    if split_name == "train":
        return [int(x) for x in splits.get("train", [])]
    sub = splits.get("train_subsets") or {}
    if split_name in sub:
        return [int(x) for x in sub[split_name]]
    raise typer.BadParameter(f"unknown split {split_name!r}")


def _load_sae(sae_path: Path, device: str) -> tuple[TopKSAE, np.ndarray, np.ndarray]:
    payload = torch.load(sae_path, map_location=device, weights_only=False)
    sae = TopKSAE(
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        k=int(payload["k"]),
    ).to(device)
    sae.load_state_dict(payload["state_dict"], strict=True)
    sae.eval()
    cls_mean = np.asarray(payload["cls_mean"], dtype=np.float32)
    cls_std = np.asarray(payload["cls_std"], dtype=np.float32)
    return sae, cls_mean, cls_std


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    splits_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/splits.yaml"), "--splits-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained SAE directory (containing sae.pt). Default: latest "
             "under checkpoints/sae/.",
    ),
    train_split: str = typer.Option("train_50k", "--train-split"),
    n_specimens: int = typer.Option(
        10000, "--n-specimens",
        help="Number of specimens used to compute feature statistics. "
             "10K is plenty.",
    ),
    top_n: int = typer.Option(
        50, "--top-n",
        help="Top activating specimens summarized per feature.",
    ),
    min_purity: float = typer.Option(0.70, "--min-purity"),
    min_corr: float = typer.Option(0.30, "--min-corr"),
    out: Path = typer.Option(Path("runs/sae_labels"), "--out", "-o"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Label every SAE feature using ground-truth attribute correlations."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if sae_dir is None:
        sae_dir = _latest_dir(Path("checkpoints/sae"))
        if sae_dir is None:
            raise typer.BadParameter(
                "no SAE under checkpoints/sae/. Run scripts/train_sae.sh first."
            )
    sae_path = sae_dir / "sae.pt"
    if not sae_path.exists():
        raise typer.BadParameter(f"no sae.pt under {sae_dir}")

    fm2_ckpt = _latest_fm2(Path("checkpoints"), train_split)

    run_id = generate_run_id("sae-labels")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id   : {run_id}")
    typer.echo(f"==> Output   : {out_dir}")
    typer.echo(f"==> SAE      : {sae_path}")
    typer.echo(f"==> FM2      : {fm2_ckpt}")

    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    sae, cls_mean, cls_std = _load_sae(sae_path, device=device)
    typer.echo(
        f"==> SAE config: in_dim={sae.in_dim} hidden_dim={sae.hidden_dim} k={sae.k}"
    )

    pool = _load_specimen_ids(splits_path, train_split)
    pool = pool[: max(n_specimens, 1)]
    typer.echo(f"==> Specimens: {len(pool)}")

    typer.echo("==> Forward FM2 + SAE...")
    activations: list[np.ndarray] = []
    motifs: list[str] = []
    atom_counts: list[int] = []
    temperatures: list[float] = []

    with h5py.File(h5_path, "r") as f, torch.no_grad():
        motif_names: list[str] = []
        if "motif_names" in f.attrs:
            motif_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f.attrs["motif_names"]
            ]

        cls_mean_t = torch.from_numpy(cls_mean).to(device)
        cls_std_t = torch.from_numpy(cls_std).to(device)

        for start in range(0, len(pool), 256):
            batch_ids = pool[start : start + 256]
            rdfs_np = np.stack(
                [np.asarray(f["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device)
            hidden = fm2.encode(rdfs)
            cls = hidden[:, 0, :]
            cls_norm = (cls - cls_mean_t) / cls_std_t
            z = sae.encode(cls_norm)
            activations.append(z.detach().cpu().numpy())
            for sid in batch_ids:
                mid = int(np.asarray(f["motif_ids"][sid]))
                motif = (
                    motif_names[mid] if 0 <= mid < len(motif_names) else str(mid)
                )
                motifs.append(motif)
                atom_counts.append(int(np.asarray(f["atom_counts"][sid])))
                temperatures.append(float(np.asarray(f["temperatures"][sid])))

    activations_arr = np.concatenate(activations, axis=0)   # (N, hidden_dim)
    motifs_arr = np.asarray(motifs)
    atom_counts_arr = np.asarray(atom_counts, dtype=np.float32)
    temperatures_arr = np.asarray(temperatures, dtype=np.float32)
    phases_arr = np.asarray([_phase_for(float(t)) for t in temperatures])

    typer.echo(
        f"    activations shape: {activations_arr.shape}, "
        f"mean active fraction: {(activations_arr > 0).mean():.4f}"
    )

    typer.echo("==> Labelling features...")
    labels_str: dict[int, str] = {}
    details_records: list[dict] = []
    n_categorical = 0
    n_continuous = 0
    n_unlabelled = 0
    for i in range(activations_arr.shape[1]):
        feat = activations_arr[:, i]
        rec = label_feature(
            feature_idx=i,
            feature_activations=feat,
            motifs=motifs_arr,
            atom_counts=atom_counts_arr,
            temperatures=temperatures_arr,
            phases=phases_arr,
            top_n=top_n,
            min_purity=min_purity,
            min_corr=min_corr,
        )
        labels_str[i] = rec.label
        details_records.append(asdict(rec))
        if rec.motif_top is not None or rec.phase_top is not None:
            n_categorical += 1
        elif rec.tags:
            n_continuous += 1
        else:
            n_unlabelled += 1

    typer.echo(
        f"    categorical-locked: {n_categorical} / "
        f"continuous-only: {n_continuous} / "
        f"unlabelled: {n_unlabelled}"
    )

    labels_path = out_dir / "labels.json"
    with labels_path.open("w") as f:
        json.dump(labels_str, f, indent=2)
    typer.echo(f"==> Labels   : {labels_path}")

    details_path = out_dir / "details.yaml"
    with details_path.open("w") as f:
        yaml.safe_dump(
            {"features": details_records},
            f,
            sort_keys=False,
        )
    typer.echo(f"==> Details  : {details_path}")

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.label_sae_features",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "sae_path": str(sae_path),
            "fm2_checkpoint": str(fm2_ckpt),
            "train_split": train_split,
            "n_specimens": len(pool),
        },
        config={
            "run_id": run_id,
            "top_n": top_n,
            "min_purity": min_purity,
            "min_corr": min_corr,
        },
        extra={
            "n_features": int(activations_arr.shape[1]),
            "n_categorical_locked": n_categorical,
            "n_continuous_only": n_continuous,
            "n_unlabelled": n_unlabelled,
        },
    )


if __name__ == "__main__":
    app()
