"""CLI: label every Qwen-SAE feature against task and specimen attributes.

Phase 15 Stage C. Loads:

  * the trained Qwen SAE (Stage B output),
  * the harvested activations + per-row metadata (Stage A output),

forwards each row through the SAE encoder to get sparse latents, and
labels every feature using categorical locks against verdict /
correctness / motif / phase plus continuous correlations against
atom count and temperature.

Output:

    runs/qwen_sae_labels/<run_id>/labels.json     # feature_idx -> label string
    runs/qwen_sae_labels/<run_id>/details.yaml    # full LLMFeatureLabel records
    runs/qwen_sae_labels/<run_id>/steering_candidates.yaml
            # ranked candidates for Stage D ablation, organized by
            # target axis ("wrong-PASS" features, "caveat" features,
            # etc.)
    runs/qwen_sae_labels/<run_id>/manifest.yaml

Usage:

    bash scripts/label_qwen_sae_features.sh

Depends on:
    typer, torch, numpy, pyyaml.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.representation.llm_labels import (  # noqa: E402
    label_llm_feature,
    rank_features_for_steering,
)
from fmllm.representation.sae import TopKSAE  # noqa: E402
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


def _load_sae(
    sae_path: Path, device: str,
) -> tuple[TopKSAE, np.ndarray, np.ndarray]:
    payload = torch.load(sae_path, map_location=device, weights_only=False)
    sae = TopKSAE(
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        k=int(payload["k"]),
    ).to(device)
    sae.load_state_dict(payload["state_dict"], strict=True)
    sae.eval()
    cls_mean = np.asarray(payload["cls_mean"], dtype=np.float32).reshape(-1)
    cls_std = np.asarray(payload["cls_std"], dtype=np.float32).reshape(-1)
    return sae, cls_mean, cls_std


def _load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r") as f:
        return yaml.safe_load(f)


@app.command()
def main(
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained Qwen SAE directory. Default: latest under "
             "checkpoints/qwen_sae/.",
    ),
    activations_dir: Path | None = typer.Option(
        None, "--activations-dir",
        help="Harvested activations directory. Default: latest under "
             "runs/qwen_activations/.",
    ),
    out: Path = typer.Option(Path("runs/qwen_sae_labels"), "--out", "-o"),
    top_n: int = typer.Option(50, "--top-n"),
    min_purity: float = typer.Option(0.70, "--min-purity"),
    min_corr: float = typer.Option(0.30, "--min-corr"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Label every Qwen-SAE feature using verdict/correctness/specimen attrs."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if sae_dir is None:
        sae_dir = _latest_dir(Path("checkpoints/qwen_sae"))
        if sae_dir is None:
            raise typer.BadParameter(
                "no Qwen SAE under checkpoints/qwen_sae/. "
                "Run scripts/train_qwen_sae.sh first."
            )
    sae_path = sae_dir / "sae.pt"
    if not sae_path.exists():
        raise typer.BadParameter(f"no sae.pt under {sae_dir}")

    if activations_dir is None:
        activations_dir = _latest_dir(Path("runs/qwen_activations"))
        if activations_dir is None:
            raise typer.BadParameter(
                "no harvested activations under runs/qwen_activations/."
            )
    acts_path = activations_dir / "activations.npy"
    metadata_path = activations_dir / "metadata.yaml"
    if not acts_path.exists() or not metadata_path.exists():
        raise typer.BadParameter(
            f"missing activations.npy or metadata.yaml under {activations_dir}"
        )

    run_id = generate_run_id("qwen-sae-labels")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> SAE         : {sae_path}")
    typer.echo(f"==> Activations : {acts_path}")
    typer.echo(f"==> Metadata    : {metadata_path}")

    sae, cls_mean, cls_std = _load_sae(sae_path, device=device)
    typer.echo(
        f"==> SAE config  : in_dim={sae.in_dim} hidden_dim={sae.hidden_dim} k={sae.k}"
    )

    acts = np.load(acts_path).astype(np.float32)
    if acts.ndim != 2:
        raise typer.BadParameter(f"unexpected activations shape: {acts.shape}")
    typer.echo(f"    rows        : {acts.shape[0]}")

    md = _load_metadata(metadata_path)
    rows: list[dict] = list(md.get("rows", []))
    if len(rows) != acts.shape[0]:
        raise typer.BadParameter(
            f"metadata rows ({len(rows)}) != activations rows "
            f"({acts.shape[0]})"
        )

    # Per-row attribute arrays in row order.
    verdicts = np.array(
        [str(r.get("verdict", "null")) for r in rows], dtype=object,
    )
    is_correct = np.array(
        [bool(r.get("is_correct", False)) for r in rows],
    )
    motifs = np.array(
        [str(r.get("ground_truth", {}).get("motif", "?")) for r in rows],
        dtype=object,
    )
    phases = np.array(
        [str(r.get("ground_truth", {}).get("phase", "?")) for r in rows],
        dtype=object,
    )
    atom_counts = np.array(
        [float(r.get("ground_truth", {}).get("n_atoms", 0)) for r in rows],
        dtype=np.float32,
    )
    temperatures = np.array(
        [float(r.get("ground_truth", {}).get("temperature", 0.0)) for r in rows],
        dtype=np.float32,
    )

    # Forward acts through the SAE encoder (with normalization) to get
    # sparse latent activations.
    typer.echo("==> Forward SAE encoder over harvested rows...")
    cls_mean_t = torch.from_numpy(cls_mean).to(device)
    cls_std_t = torch.from_numpy(cls_std).to(device)
    with torch.no_grad():
        x = torch.from_numpy(acts).to(device)
        x_norm = (x - cls_mean_t) / cls_std_t.clamp_min(1.0e-6)
        z = sae.encode(x_norm).detach().cpu().numpy()      # (N, hidden_dim)
    typer.echo(
        f"    activations shape: {z.shape}, "
        f"mean active fraction: {(z > 0).mean():.4f}"
    )

    # Label every feature ------------------------------------------------
    typer.echo("==> Labelling features...")
    labels_str: dict[int, str] = {}
    details: list[dict] = []
    label_objs = []
    n_locked = 0
    n_unlabelled = 0
    n_rare = 0
    for i in range(z.shape[1]):
        feat = z[:, i]
        rec = label_llm_feature(
            feature_idx=i,
            feature_activations=feat,
            verdicts=verdicts,
            is_correct=is_correct,
            motifs=motifs,
            phases=phases,
            atom_counts=atom_counts,
            temperatures=temperatures,
            top_n=top_n,
            min_purity=min_purity,
            min_corr=min_corr,
        )
        label_objs.append(rec)
        labels_str[i] = rec.label
        details.append(asdict(rec))
        if "(rare)" in rec.label:
            n_rare += 1
        elif rec.tags:
            n_locked += 1
        else:
            n_unlabelled += 1

    typer.echo(
        f"    locked: {n_locked} | unlabelled: {n_unlabelled} | rare: {n_rare}"
    )

    # Persist labels -----------------------------------------------------
    labels_path = out_dir / "labels.json"
    with labels_path.open("w") as f:
        json.dump(labels_str, f, indent=2)
    typer.echo(f"==> Labels      : {labels_path}")

    details_path = out_dir / "details.yaml"
    with details_path.open("w") as f:
        yaml.safe_dump({"features": details}, f, sort_keys=False)
    typer.echo(f"==> Details     : {details_path}")

    # Steering candidates ------------------------------------------------
    candidates: dict[str, list[dict]] = {
        "wrong_pass": [
            asdict(l) for l in rank_features_for_steering(
                label_objs, target_axis="correct", target_value=False,
                min_purity=min_purity,
            )
            if l.verdict_top == "pass"
        ],
        "wrong_any": [
            asdict(l) for l in rank_features_for_steering(
                label_objs, target_axis="correct", target_value=False,
                min_purity=min_purity,
            )
        ],
        "caveat": [
            asdict(l) for l in rank_features_for_steering(
                label_objs, target_axis="verdict", target_value="caveat",
                min_purity=min_purity,
            )
        ],
    }
    steering_path = out_dir / "steering_candidates.yaml"
    with steering_path.open("w") as f:
        yaml.safe_dump(candidates, f, sort_keys=False)
    typer.echo(f"==> Candidates  : {steering_path}")
    typer.echo(
        f"    wrong-PASS: {len(candidates['wrong_pass'])}  "
        f"wrong-any : {len(candidates['wrong_any'])}  "
        f"caveat    : {len(candidates['caveat'])}"
    )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.label_qwen_sae_features",
        inputs={
            "sae_path": str(sae_path),
            "activations_dir": str(activations_dir),
        },
        config={
            "run_id": run_id,
            "top_n": top_n,
            "min_purity": min_purity,
            "min_corr": min_corr,
        },
        extra={
            "n_features": int(z.shape[1]),
            "n_rows": int(z.shape[0]),
            "n_locked": n_locked,
            "n_unlabelled": n_unlabelled,
            "n_rare": n_rare,
            "n_wrong_pass_candidates": len(candidates["wrong_pass"]),
            "n_caveat_candidates": len(candidates["caveat"]),
        },
    )


if __name__ == "__main__":
    app()
