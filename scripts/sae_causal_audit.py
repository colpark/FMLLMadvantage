"""CLI: causal audit of SAE features against FM2's energy head.

Phase 14. For every feature in the trained Top-K SAE, run a
counterfactual intervention on the audit set and measure the resulting
change in FM2's predicted per-atom energy. Output a per-feature
causal record plus a filter list of feature indices that pass a
configurable causal-effect threshold.

Two questions this answers:

  1. Are the SAE features we feed the LLM actually used by FM2's
     downstream head, or are they decorative directions in the
     representation that have no causal role in the prediction?
  2. Which subset of features carries enough signal that the LLM
     prompt should mention them, vs which should be dropped?

Output:

    runs/sae_causal/<run_id>/causal_effects.yaml   # per-feature record
    runs/sae_causal/<run_id>/causal_filter.json    # passing feature ids
    runs/sae_causal/<run_id>/manifest.yaml

Usage:

    bash scripts/sae_causal_audit.sh
    uv run python scripts/sae_causal_audit.py --min-norm-effect 0.10

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

from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.representation.causal import (  # noqa: E402
    audit_feature,
    filter_features_by_causal_effect,
)
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


def _load_sae(
    sae_path: Path, device: str,
) -> tuple[TopKSAE, torch.Tensor, torch.Tensor]:
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


def _extract_cls_batch(
    *,
    fm2: torch.nn.Module,
    h5_path: Path,
    specimen_ids: list[int],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Forward FM2 over a list of specimens, return un-normalized CLS."""
    fm2.eval()
    blocks: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f, torch.no_grad():
        for start in range(0, len(specimen_ids), batch_size):
            batch_ids = specimen_ids[start : start + batch_size]
            rdfs_np = np.stack(
                [np.asarray(f["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device).float()
            hidden = fm2.encode(rdfs)
            cls = hidden[:, 0, :].detach().cpu().numpy()
            blocks.append(cls)
    arr = np.concatenate(blocks, axis=0)
    return torch.from_numpy(arr).to(device)


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
        help="Trained SAE directory. Default: latest under checkpoints/sae/.",
    ),
    sae_labels_path: Path | None = typer.Option(
        None, "--sae-labels-path",
        help="labels.json for the SAE. Default: latest under runs/sae_labels/.",
    ),
    train_split: str = typer.Option("train_50k", "--train-split"),
    n_specimens: int = typer.Option(
        2000, "--n-specimens",
        help="Number of specimens used for the causal audit. 2K is "
             "plenty for an effect-size estimate; the audit complexity "
             "is O(hidden_dim * n_specimens) so this scales linearly.",
    ),
    feature_subset: str | None = typer.Option(
        None, "--feature-subset",
        help="Comma-separated feature indices to audit. Default: all "
             "features in the SAE (hidden_dim).",
    ),
    min_norm_effect: float = typer.Option(
        0.10, "--min-norm-effect",
        help="Threshold on |knock-out or knock-in effect| / std(recon "
             "energy) for a feature to pass the causal filter.",
    ),
    min_activation_rate: float = typer.Option(
        0.005, "--min-activation-rate",
        help="Drop features that fire on fewer than this fraction of "
             "the audit set, regardless of effect size.",
    ),
    out: Path = typer.Option(Path("runs/sae_causal"), "--out", "-o"),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(50, "--log-every"),
) -> None:
    """Causally audit SAE features against FM2's energy head."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve artefact paths -----------------------------------------------
    if sae_dir is None:
        sae_dir = _latest_dir(Path("checkpoints/sae"))
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

    fm2_ckpt = _latest_fm2(Path("checkpoints"), train_split)

    run_id = generate_run_id("sae-causal")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id   : {run_id}")
    typer.echo(f"==> Output   : {out_dir}")
    typer.echo(f"==> SAE      : {sae_path}")
    typer.echo(f"==> Labels   : {sae_labels_path}")
    typer.echo(f"==> FM2      : {fm2_ckpt}")

    # Load SAE, FM2 -------------------------------------------------------
    sae, cls_mean, cls_std = _load_sae(sae_path, device=device)
    typer.echo(
        f"==> SAE      : in_dim={sae.in_dim} hidden_dim={sae.hidden_dim} k={sae.k}"
    )

    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    energy_head = fm2.energy_head           # used as the causal target

    # Load labels ---------------------------------------------------------
    labels = _load_labels(sae_labels_path)

    # Audit set: forward all specimens through FM2 once; reuse CLS for
    # every feature audit.
    pool = _load_specimen_ids(splits_path, train_split)
    pool = pool[: max(n_specimens, 1)]
    typer.echo(f"==> Specimens: {len(pool)}")
    typer.echo("==> Forward FM2 over audit set...")
    cls_audit = _extract_cls_batch(
        fm2=fm2, h5_path=h5_path, specimen_ids=pool,
        device=device, batch_size=256,
    )
    typer.echo(f"    cls shape: {tuple(cls_audit.shape)}")

    # Pick features to audit ----------------------------------------------
    if feature_subset:
        feature_ids = [int(x.strip()) for x in feature_subset.split(",") if x.strip()]
    else:
        feature_ids = list(range(int(sae.hidden_dim)))
    typer.echo(f"==> Auditing {len(feature_ids)} features")

    # Audit loop ----------------------------------------------------------
    typer.echo("-" * 64)
    records: list[dict] = []
    effect_objs = []
    for j, fid in enumerate(feature_ids):
        rec = audit_feature(
            sae=sae,
            energy_head=energy_head,
            cls_original=cls_audit,
            cls_mean=cls_mean,
            cls_std=cls_std,
            feature_idx=fid,
            label=labels.get(fid, f"feature-{fid}"),
        )
        effect_objs.append(rec)
        records.append(asdict(rec))
        if (j + 1) % log_every == 0 or (j + 1) == len(feature_ids):
            typer.echo(
                f"  audited {j + 1:>5}/{len(feature_ids)} | "
                f"latest fid={fid:>4} act_rate={rec.activation_rate:.3f} "
                f"|ko_norm|={rec.knock_out_effect_norm:.3f} "
                f"|ki_norm|={rec.knock_in_effect_norm:.3f}"
            )
    typer.echo("-" * 64)

    # Filter --------------------------------------------------------------
    passing = filter_features_by_causal_effect(
        effects=effect_objs,
        min_norm_effect=min_norm_effect,
        require_activation=True,
        min_activation_rate=min_activation_rate,
    )
    n_passing = len(passing)
    typer.echo(
        f"==> Causal filter: {n_passing}/{len(feature_ids)} features pass "
        f"(min_norm_effect={min_norm_effect}, "
        f"min_activation_rate={min_activation_rate})"
    )

    # Persist -------------------------------------------------------------
    effects_path = out_dir / "causal_effects.yaml"
    with effects_path.open("w") as f:
        yaml.safe_dump(
            {
                "config": {
                    "min_norm_effect": min_norm_effect,
                    "min_activation_rate": min_activation_rate,
                    "n_specimens": len(pool),
                },
                "features": records,
            },
            f,
            sort_keys=False,
        )
    typer.echo(f"==> Effects  : {effects_path}")

    filter_path = out_dir / "causal_filter.json"
    with filter_path.open("w") as f:
        json.dump(
            {
                "passing_feature_ids": passing,
                "min_norm_effect": min_norm_effect,
                "min_activation_rate": min_activation_rate,
                "n_audited": len(feature_ids),
                "n_passing": n_passing,
            },
            f,
            indent=2,
        )
    typer.echo(f"==> Filter   : {filter_path}")

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.sae_causal_audit",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "sae_path": str(sae_path),
            "sae_labels_path": str(sae_labels_path),
            "fm2_checkpoint": str(fm2_ckpt),
            "train_split": train_split,
            "n_specimens": len(pool),
        },
        config={
            "run_id": run_id,
            "min_norm_effect": min_norm_effect,
            "min_activation_rate": min_activation_rate,
        },
        extra={
            "n_features_audited": len(feature_ids),
            "n_features_passing": n_passing,
        },
    )


if __name__ == "__main__":
    app()
