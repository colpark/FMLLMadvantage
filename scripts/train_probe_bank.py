"""CLI: train a probe bank on top of frozen FM2 features.

Phase 11 Stage 0 prerequisite. Each probe is a small head trained
against a structured label derived from the dataset HDF5 (or simple
geometry over equilibrium positions). The probes the CoT generator
expects:

    - n_atoms          : regression on int atom count
    - motif            : 3-class classification (triangular_disk, ring, linear)
    - phase            : 3-class classification (solid-like, liquid-like, gas-like)
    - coordination     : regression on mean first-shell coordination
    - peak_position    : regression on the RDF first-peak position (~1.13 LJ)

After training, the bank is saved as a directory of ``.pt`` files
plus ``manifest.yaml`` and consumed by ``scripts/build_cot_dataset.py``
and any inference-time probe consumer.

Usage:

    bash scripts/train_probe_bank.sh
    uv run python scripts/train_probe_bank.py --epochs 50

Depends on:
    typer, torch, h5py, pyyaml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.connectors.text_annotations import (  # noqa: E402
    _cluster_diameter,
    _mean_coordination,
    _phase_for,
)
from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.training.probe_bank import ProbeBank, ProbeSpec, _build_module  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


_PHASE_TO_LABEL = {"solid-like": 0, "liquid-like": 1, "gas-like": 2}
_LABEL_TO_PHASE = {v: k for k, v in _PHASE_TO_LABEL.items()}


def _latest_supervised_fm2(
    checkpoint_root: Path, train_split: str,
) -> Path:
    cands = sorted(
        (checkpoint_root / "fm2_rdf" / train_split).glob("*"),
        key=lambda p: p.name, reverse=True,
    )
    cands = [c for c in cands if (c / "model.pt").exists()]
    if not cands:
        raise typer.BadParameter(
            f"no completed fm2_rdf checkpoint under "
            f"{checkpoint_root}/fm2_rdf/{train_split}/"
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


def _extract_features(
    *,
    model: torch.nn.Module,
    h5_path: Path,
    specimen_ids: list[int],
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    cls_blocks: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f, torch.no_grad():
        for start in range(0, len(specimen_ids), batch_size):
            batch_ids = specimen_ids[start : start + batch_size]
            rdfs_np = np.stack(
                [np.asarray(f["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device).float()
            hidden = model.encode(rdfs)               # (B, T, D)
            cls = hidden[:, 0, :].detach().cpu().numpy()
            cls_blocks.append(cls)
    return np.concatenate(cls_blocks, axis=0)


def _build_labels(
    *,
    h5_path: Path,
    specimen_ids: list[int],
) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        atom_counts = np.asarray(f["atom_counts"])
        temperatures = np.asarray(f["temperatures"])
        motif_ids = np.asarray(f["motif_ids"])
        eq = np.asarray(f["equilibrium_positions"])
        rdfs = np.asarray(f["rdfs"])
        rdf_min = float(f.attrs.get("rdf_r_min", 0.0))
        rdf_max = float(f.attrs.get("rdf_r_max", 5.0))
        motif_names: list[str] = []
        if "motif_names" in f.attrs:
            motif_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f.attrs["motif_names"]
            ]

    n_arr = []
    motif_arr = []
    phase_arr = []
    coord_arr = []
    peak_arr = []
    bin_centers = np.linspace(rdf_min, rdf_max, rdfs.shape[1])

    for sid in specimen_ids:
        n = int(atom_counts[sid])
        t = float(temperatures[sid])
        mid = int(motif_ids[sid])
        motif = (
            motif_names[mid] if mid < len(motif_names) else str(mid)
        )
        positions = np.asarray(eq[sid])[:n]
        coord = (
            float(_mean_coordination(positions, cutoff=1.4))
            if positions.size > 0 else 0.0
        )
        rdf = np.asarray(rdfs[sid])
        peak_idx = int(np.argmax(rdf))
        peak_pos = float(bin_centers[peak_idx])
        n_arr.append(float(n))
        motif_arr.append(motif)
        phase_arr.append(_phase_for(t))
        coord_arr.append(coord)
        peak_arr.append(peak_pos)

    motif_index = {m: i for i, m in enumerate(motif_names)}
    return {
        "n_atoms": np.asarray(n_arr, dtype=np.float32),
        "motif_idx": np.asarray(
            [motif_index[m] for m in motif_arr], dtype=np.int64,
        ),
        "motif_names": motif_names,
        "phase_idx": np.asarray(
            [_PHASE_TO_LABEL[p] for p in phase_arr], dtype=np.int64,
        ),
        "coordination": np.asarray(coord_arr, dtype=np.float32),
        "peak_position": np.asarray(peak_arr, dtype=np.float32),
    }


def _train_probe(
    *,
    spec: ProbeSpec,
    features: np.ndarray,
    labels_train: np.ndarray,
    labels_eval: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    epochs: int,
    lr: float,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    module = _build_module(spec).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1.0e-4)

    X_tr = torch.from_numpy(features[train_idx]).float().to(device)
    X_ev = torch.from_numpy(features[eval_idx]).float().to(device)
    if spec.kind == "regression":
        y_tr = torch.from_numpy(labels_train).float().to(device)
        y_ev = torch.from_numpy(labels_eval).float().to(device)
        loss_fn = torch.nn.MSELoss()
    else:
        y_tr = torch.from_numpy(labels_train).long().to(device)
        y_ev = torch.from_numpy(labels_eval).long().to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        module.train()
        opt.zero_grad()
        pred = module(X_tr)
        if spec.kind == "regression":
            loss = loss_fn(pred.squeeze(-1), y_tr)
        else:
            loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()

    module.eval()
    with torch.no_grad():
        pred_ev = module(X_ev)
        if spec.kind == "regression":
            mse = float(torch.nn.functional.mse_loss(pred_ev.squeeze(-1), y_ev).item())
            mae = float((pred_ev.squeeze(-1) - y_ev).abs().mean().item())
            var = float(y_ev.var().item())
            r2 = 1.0 - mse / max(var, 1.0e-9)
            metrics = {"mse": mse, "mae": mae, "r2": r2}
        else:
            preds = pred_ev.argmax(dim=-1)
            acc = float((preds == y_ev).float().mean().item())
            metrics = {"accuracy": acc}
    return module, metrics


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
    n_specimens: int = typer.Option(
        10000, "--n-specimens",
        help="Number of specimens used to train the probes. 10K is "
             "usually plenty since each probe is small.",
    ),
    out: Path = typer.Option(
        Path("checkpoints/probes"), "--out", "-o",
    ),
    epochs: int = typer.Option(50, "--epochs"),
    lr: float = typer.Option(1.0e-3, "--lr"),
    hidden: int = typer.Option(128, "--hidden"),
    eval_frac: float = typer.Option(0.20, "--eval-frac"),
    seed: int = typer.Option(0, "--seed"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Train a probe bank on FM2 features."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_id = generate_run_id("probe-bank")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fm2_ckpt = _latest_supervised_fm2(checkpoint_root, train_split)
    typer.echo(f"==> Run id   : {run_id}")
    typer.echo(f"==> Output   : {out_dir}")
    typer.echo(f"==> FM2 ckpt : {fm2_ckpt}")

    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    pool = _load_specimen_ids(splits_path, train_split)
    if not pool:
        raise typer.BadParameter(f"empty split {train_split!r}")
    pool = pool[: max(n_specimens, 1)]
    typer.echo(f"==> Probe set: {len(pool)} specimens")

    typer.echo("==> Extracting FM2 features...")
    features = _extract_features(
        model=fm2,
        h5_path=h5_path,
        specimen_ids=pool,
        device=device,
        batch_size=256,
    )
    in_dim = int(features.shape[1])
    typer.echo(f"    feature shape: {features.shape}")

    typer.echo("==> Building labels...")
    labels = _build_labels(h5_path=h5_path, specimen_ids=pool)
    motif_names: list[str] = list(labels["motif_names"])

    n = features.shape[0]
    perm = np.random.permutation(n)
    n_eval = max(1, int(eval_frac * n))
    eval_idx = perm[:n_eval]
    train_idx = perm[n_eval:]

    bank = ProbeBank()
    metrics: dict[str, Any] = {}

    # Probe definitions ---------------------------------------------------
    probes_to_train: list[ProbeSpec] = [
        ProbeSpec(
            name="n_atoms", kind="regression", in_dim=in_dim,
            out_dim=1, hidden=hidden,
            target_min=float(labels["n_atoms"].min()),
            target_max=float(labels["n_atoms"].max()),
            target_mean=float(labels["n_atoms"].mean()),
            target_std=float(labels["n_atoms"].std()),
        ),
        ProbeSpec(
            name="motif", kind="classification", in_dim=in_dim,
            out_dim=len(motif_names), hidden=hidden,
            class_names=motif_names,
        ),
        ProbeSpec(
            name="phase", kind="classification", in_dim=in_dim,
            out_dim=len(_PHASE_TO_LABEL), hidden=hidden,
            class_names=[_LABEL_TO_PHASE[i] for i in range(len(_PHASE_TO_LABEL))],
        ),
        ProbeSpec(
            name="coordination", kind="regression", in_dim=in_dim,
            out_dim=1, hidden=hidden,
            target_min=float(labels["coordination"].min()),
            target_max=float(labels["coordination"].max()),
            target_mean=float(labels["coordination"].mean()),
            target_std=float(labels["coordination"].std()),
        ),
        ProbeSpec(
            name="peak_position", kind="regression", in_dim=in_dim,
            out_dim=1, hidden=hidden,
            target_min=float(labels["peak_position"].min()),
            target_max=float(labels["peak_position"].max()),
            target_mean=float(labels["peak_position"].mean()),
            target_std=float(labels["peak_position"].std()),
        ),
    ]

    label_keys: dict[str, str] = {
        "n_atoms": "n_atoms",
        "motif": "motif_idx",
        "phase": "phase_idx",
        "coordination": "coordination",
        "peak_position": "peak_position",
    }

    typer.echo("")
    typer.echo("==> Training probes")
    typer.echo("-" * 60)
    for spec in probes_to_train:
        label = labels[label_keys[spec.name]]
        module, m = _train_probe(
            spec=spec,
            features=features,
            labels_train=label[train_idx],
            labels_eval=label[eval_idx],
            train_idx=train_idx,
            eval_idx=eval_idx,
            epochs=epochs,
            lr=lr,
            device=device,
        )
        bank.add(spec, module)
        metrics[spec.name] = m
        if spec.kind == "regression":
            typer.echo(
                f"  {spec.name:<14} r2={m['r2']:.4f} mae={m['mae']:.4f}"
            )
        else:
            typer.echo(
                f"  {spec.name:<14} acc={m['accuracy']:.4f}"
            )
    typer.echo("-" * 60)

    typer.echo(f"==> Saving probe bank to {out_dir}")
    bank.save(out_dir)

    write_manifest(
        out_dir / "manifest_run.yaml",
        script="scripts.train_probe_bank",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "fm2_checkpoint": str(fm2_ckpt),
            "train_split": train_split,
            "n_specimens": len(pool),
        },
        config={
            "run_id": run_id,
            "epochs": epochs,
            "lr": lr,
            "hidden": hidden,
            "eval_frac": eval_frac,
            "seed": seed,
            "feature_dim": in_dim,
        },
        extra={
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    app()
