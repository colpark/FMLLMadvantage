"""CLI: Phase 9.0 probing study on FM2's frozen backbone.

Question: does FM2's energy-supervised representation hold task-extra
signal that the energy head does not surface?

Method: freeze FM2 from a trained checkpoint, extract the CLS
embedding for every specimen in a chosen split, train a small probe
(linear or 2-layer MLP) on each ground-truth target, report metrics
on a held-out portion of the same split.

Targets:
    - atom_count          (regression, sanity probe)
    - cluster_diameter    (regression, geometric)
    - mean_coordination   (regression, structural)
    - phase               (3-class classification, thermodynamic)

Decision rule for Phase 9.A:
    All probes >= 0.85 score                     ⇒  representation is rich,
                                                    proceed to connector
    Some probes succeed, others fail             ⇒  selective richness, build
                                                    connector but expect modest
                                                    gains
    All probes near chance                       ⇒  representation collapsed
                                                    to energy, skip connector,
                                                    consider self-supervised
                                                    pretraining (Layer D)

Output:
    runs/probes/<run_id>/report.yaml
    runs/probes/<run_id>/manifest.yaml

Usage:
    uv run python scripts/run_fm2_probes.py
    uv run python scripts/run_fm2_probes.py \\
        --train-split train_50k --probe-split val \\
        --probe-arch mlp --epochs 30

Depends on:
    typer, torch, h5py, numpy, scikit-learn (lazy).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.connectors.text_annotations import (  # noqa: E402
    annotate_specimen_from_h5,
    annotation_label_dict,
)
from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.fms.fm2_rdf_ssl.model import build_fm2_ssl_model  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


_PHASE_TO_LABEL = {"solid-like": 0, "liquid-like": 1, "gas-like": 2}


def _latest_checkpoint_dir(
    checkpoint_root: Path, train_split: str, kind: str = "fm2_rdf",
) -> Path:
    """Find the latest run-id under ``checkpoints/<kind>/<train_split>/``.

    ``kind`` is ``fm2_rdf`` for the supervised backbone, or
    ``fm2_rdf_ssl`` for the masked-RDF backbone produced by Phase 10.
    """
    candidates = sorted(
        (checkpoint_root / kind / train_split).glob("*"),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise typer.BadParameter(
            f"no {kind} checkpoint under "
            f"{checkpoint_root}/{kind}/{train_split}/"
        )
    return candidates[0]


def _load_split_ids(splits_path: Path, split_name: str) -> list[int]:
    with splits_path.open("r") as f:
        splits = yaml.safe_load(f)
    if split_name in ("train", "train_50k", "train_30k", "train_10k"):
        if split_name == "train":
            return [int(x) for x in splits.get("train", [])]
        subsets = splits.get("train_subsets") or {}
        if split_name in subsets:
            return [int(x) for x in subsets[split_name]]
    if split_name == "val":
        # If a dedicated 'val' partition exists, use it; otherwise fall
        # back to a slice of `train` we will sub-split below.
        if isinstance(splits.get("val"), list):
            return [int(x) for x in splits["val"]]
    if split_name == "holdout":
        ho = splits.get("holdout")
        if isinstance(ho, dict):
            ids: list[int] = []
            for cell in ho.values():
                if isinstance(cell, list):
                    ids.extend(int(x) for x in cell)
            return sorted(ids)
    raise typer.BadParameter(f"unknown split {split_name!r}")


def _extract_features(
    *,
    model: torch.nn.Module,
    h5_path: Path,
    specimen_ids: list[int],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Run frozen FM2 in encode() mode and return ``(N, embed_dim)`` CLS
    embeddings for the requested specimens."""
    import h5py  # noqa: PLC0415

    model.eval()
    with torch.no_grad():
        cls_blocks: list[np.ndarray] = []
        with h5py.File(Path(h5_path), "r") as f:
            for start in range(0, len(specimen_ids), batch_size):
                batch_ids = specimen_ids[start : start + batch_size]
                rdfs_np = np.stack(
                    [np.asarray(f["rdfs"][i]) for i in batch_ids], axis=0,
                )
                rdfs = torch.from_numpy(rdfs_np).float().to(device)
                hidden = model.encode(rdfs)              # (B, T, D)
                cls = hidden[:, 0, :].detach().cpu().numpy()
                cls_blocks.append(cls)
    return np.concatenate(cls_blocks, axis=0)


def _build_labels(
    *, h5_path: Path, specimen_ids: list[int], use_positions: bool,
) -> dict[str, np.ndarray]:
    """Generate per-specimen ground-truth labels for every probe target."""
    n_atoms: list[float] = []
    diameter: list[float] = []
    coord: list[float] = []
    phase: list[int] = []
    for sid in specimen_ids:
        ann = annotate_specimen_from_h5(
            h5_path, sid, use_positions=use_positions,
        )
        labels = annotation_label_dict(ann)
        n_atoms.append(float(labels["n_atoms"]))
        diameter.append(
            float(labels["diameter_lj"])
            if labels["diameter_lj"] is not None else float("nan")
        )
        coord.append(
            float(labels["mean_coordination"])
            if labels["mean_coordination"] is not None else float("nan")
        )
        phase.append(_PHASE_TO_LABEL[labels["phase"]])
    return {
        "n_atoms": np.asarray(n_atoms, dtype=np.float32),
        "diameter_lj": np.asarray(diameter, dtype=np.float32),
        "mean_coordination": np.asarray(coord, dtype=np.float32),
        "phase": np.asarray(phase, dtype=np.int64),
    }


def _split_train_eval(
    n: int, eval_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random index split for training the probe and reporting metrics."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_eval = max(1, int(eval_frac * n))
    return idx[n_eval:], idx[:n_eval]


def _train_regression_probe(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    arch: str,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Train a probe and return metrics on the eval split."""
    mask = ~np.isnan(targets)
    if mask.sum() < 10:
        return {"skipped": True, "reason": "too few non-nan labels"}
    X = features[mask]
    y = targets[mask]

    train_idx, eval_idx = _split_train_eval(X.shape[0], eval_frac=0.2, seed=seed)
    X_tr, X_ev = X[train_idx], X[eval_idx]
    y_tr, y_ev = y[train_idx], y[eval_idx]

    Xt_tr = torch.from_numpy(X_tr).float().to(device)
    yt_tr = torch.from_numpy(y_tr).float().to(device)
    Xt_ev = torch.from_numpy(X_ev).float().to(device)
    yt_ev = torch.from_numpy(y_ev).float().to(device)

    in_dim = X.shape[1]
    if arch == "linear":
        probe = torch.nn.Linear(in_dim, 1).to(device)
    elif arch == "mlp":
        probe = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1),
        ).to(device)
    else:
        raise typer.BadParameter(f"unknown probe arch {arch!r}")

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1.0e-4)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        pred = probe(Xt_tr).squeeze(-1)
        loss = loss_fn(pred, yt_tr)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred_ev = probe(Xt_ev).squeeze(-1)
        mse = torch.nn.functional.mse_loss(pred_ev, yt_ev).item()
        mae = (pred_ev - yt_ev).abs().mean().item()
        var = float(yt_ev.var().item())
        r2 = 1.0 - mse / max(var, 1.0e-9)

    return {
        "skipped": False,
        "n_train": int(X_tr.shape[0]),
        "n_eval": int(X_ev.shape[0]),
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
    }


def _train_classification_probe(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    arch: str,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if features.shape[0] < 10:
        return {"skipped": True, "reason": "too few specimens"}
    train_idx, eval_idx = _split_train_eval(
        features.shape[0], eval_frac=0.2, seed=seed,
    )
    X_tr, X_ev = features[train_idx], features[eval_idx]
    y_tr, y_ev = targets[train_idx], targets[eval_idx]

    Xt_tr = torch.from_numpy(X_tr).float().to(device)
    yt_tr = torch.from_numpy(y_tr).long().to(device)
    Xt_ev = torch.from_numpy(X_ev).float().to(device)
    yt_ev = torch.from_numpy(y_ev).long().to(device)

    in_dim = X_tr.shape[1]
    if arch == "linear":
        probe = torch.nn.Linear(in_dim, n_classes).to(device)
    elif arch == "mlp":
        probe = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, n_classes),
        ).to(device)
    else:
        raise typer.BadParameter(f"unknown probe arch {arch!r}")

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1.0e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(Xt_tr)
        loss = loss_fn(logits, yt_tr)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        logits_ev = probe(Xt_ev)
        pred = logits_ev.argmax(dim=-1)
        acc = float((pred == yt_ev).float().mean().item())

    return {
        "skipped": False,
        "n_train": int(X_tr.shape[0]),
        "n_eval": int(X_ev.shape[0]),
        "accuracy": float(acc),
        "n_classes": int(n_classes),
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
    probe_split: str = typer.Option(
        "train_50k", "--probe-split",
        help="Specimen split the probes train and evaluate on. The "
             "split is internally divided 80/20 train/eval. Default is "
             "train_50k for sample size; switch to a held-out slice if "
             "you have one.",
    ),
    max_specimens: int = typer.Option(
        2000, "--max-specimens",
        help="Cap on probe-split size. Probing is cheap; 2K is plenty.",
    ),
    probe_arch: str = typer.Option("mlp", "--probe-arch"),
    epochs: int = typer.Option(30, "--epochs"),
    lr: float = typer.Option(1.0e-3, "--lr"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("runs/probes"), "--out", "-o"),
    device: str = typer.Option("auto", "--device"),
    use_ssl: bool = typer.Option(
        False, "--use-ssl/--no-use-ssl",
        help="Probe the SSL backbone (Phase 10) instead of the "
             "supervised FM2. Loads from checkpoints/fm2_rdf_ssl/ "
             "and uses build_fm2_ssl_model.",
    ),
) -> None:
    """Run the FM2 probing study and write a YAML report."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone_kind = "fm2_rdf_ssl" if use_ssl else "fm2_rdf"
    slug_kind = "ssl" if use_ssl else "supervised"
    run_id = generate_run_id(f"fm2-probes-{slug_kind}-{probe_arch}")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id        : {run_id}")
    typer.echo(f"==> Output        : {out_dir}")
    typer.echo(f"==> Train split   : {train_split}")
    typer.echo(f"==> Backbone kind : {backbone_kind} ({slug_kind})")
    typer.echo(f"==> Probe split   : {probe_split}")

    # Load the requested backbone from the latest checkpoint and freeze it.
    ckpt_dir = _latest_checkpoint_dir(
        checkpoint_root, train_split, kind=backbone_kind,
    )
    model = (
        build_fm2_ssl_model(cfg.fm2) if use_ssl else build_fm2_model(cfg.fm2)
    ).to(device)
    load_checkpoint(ckpt_dir / "model.pt", model=model, map_location=device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    typer.echo(f"==> FM2 checkpoint: {ckpt_dir}")

    # Resolve specimen IDs.
    ids_all = _load_split_ids(splits_path, probe_split)
    if not ids_all:
        typer.echo(f"ERROR: probe split {probe_split!r} is empty")
        raise typer.Exit(code=1)
    if max_specimens and len(ids_all) > max_specimens:
        ids_all = ids_all[:max_specimens]
    typer.echo(f"==> Probe specimens: {len(ids_all)}")

    # Extract features and labels.
    typer.echo("==> Extracting frozen FM2 features...")
    features = _extract_features(
        model=model,
        h5_path=h5_path,
        specimen_ids=ids_all,
        device=device,
        batch_size=batch_size,
    )
    typer.echo(f"    features shape: {features.shape}")

    typer.echo("==> Building labels...")
    labels = _build_labels(
        h5_path=h5_path, specimen_ids=ids_all, use_positions=True,
    )
    for k, v in labels.items():
        typer.echo(f"    {k:<22} shape={tuple(v.shape)}")

    # Train probes.
    results: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(ckpt_dir),
        "backbone_kind": backbone_kind,
        "probe_split": probe_split,
        "n_specimens": len(ids_all),
        "feature_dim": int(features.shape[1]),
        "probe_arch": probe_arch,
        "probes": {},
    }

    typer.echo("")
    typer.echo("==> Training probes")
    typer.echo("-" * 64)
    for target in ("n_atoms", "diameter_lj", "mean_coordination"):
        m = _train_regression_probe(
            features=features,
            targets=labels[target],
            arch=probe_arch,
            epochs=epochs,
            lr=lr,
            seed=seed,
            device=device,
        )
        results["probes"][target] = m
        if m.get("skipped"):
            typer.echo(f"  {target:<22} SKIPPED ({m.get('reason')})")
        else:
            typer.echo(
                f"  {target:<22} "
                f"r2={m['r2']:.4f}  mae={m['mae']:.4f}  "
                f"n_eval={m['n_eval']}"
            )

    m_phase = _train_classification_probe(
        features=features,
        targets=labels["phase"],
        n_classes=len(_PHASE_TO_LABEL),
        arch=probe_arch,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )
    results["probes"]["phase"] = m_phase
    if m_phase.get("skipped"):
        typer.echo(f"  phase                  SKIPPED ({m_phase.get('reason')})")
    else:
        typer.echo(
            f"  {'phase':<22} "
            f"acc={m_phase['accuracy']:.4f}  n_eval={m_phase['n_eval']}"
        )
    typer.echo("-" * 64)

    # Headline interpretation.
    summary = []
    for k, v in results["probes"].items():
        if v.get("skipped"):
            continue
        if "r2" in v:
            summary.append(f"{k}={v['r2']:.3f}r2")
        else:
            summary.append(f"{k}={v['accuracy']:.3f}acc")
    typer.echo("HEADLINE: " + " | ".join(summary))
    typer.echo("")
    typer.echo(
        "Decision rule:\n"
        "  all >= 0.85   ⇒ representation rich, proceed to connector\n"
        "  mixed         ⇒ selective richness, build connector with care\n"
        "  all near chance ⇒ collapsed to energy, skip connector"
    )

    report_path = out_dir / "report.yaml"
    with report_path.open("w") as f:
        yaml.safe_dump(json.loads(json.dumps(results)), f, sort_keys=False)
    typer.echo(f"==> Report: {report_path}")

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.run_fm2_probes",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "checkpoint": str(ckpt_dir),
            "probe_split": probe_split,
            "max_specimens": max_specimens,
        },
        config={
            "probe_arch": probe_arch,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "feature_dim": int(features.shape[1]),
        },
        extra={
            "run_id": run_id,
            "device": device,
        },
    )


if __name__ == "__main__":
    app()
