"""Stage 6b (diagnostic): evaluate the trained SAE's feature usage.

Loads a trained SAE + cached embeddings, forwards every specimen
through the encoder, and reports the metrics that actually tell
you whether the SAE is healthy:

  * Dead-feature count: features that never fire on ANY specimen.
    Top-K SAEs without intervention typically have 5-15% dead
    features (Bricken et al. 2023). Anthropic's resampling recipe
    drops this to ~0%.

  * Activation-count distribution per feature: min / p10 / p50 /
    p90 / max. A flat distribution = features used uniformly (good).
    A long-tailed distribution = a few features dominate (mode
    collapse).

  * Per-specimen feature-set diversity: how many UNIQUE features
    are activated across the dataset. With k=32 and hidden=1024 we
    want a count near 1024.

  * Reconstruction error distribution: training MSE is one number;
    per-specimen MSE distribution shows whether the SAE
    reconstructs uniformly well or whether some specimens are
    poorly reconstructed (indicating they sit far from the
    feature dictionary).

The reported numbers tell you whether to upgrade to dead-feature
resampling, JumpReLU, or Gated SAE before moving to Stage 7.

Output:

    runs/materials/sae_diagnostics/<run_id>/diagnostic.yaml

Usage:

    bash scripts/materials/06b_diagnose_sae.sh

Depends on:
    typer, numpy, torch, pyyaml.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _generate_run_id(slug: str = "sae-diag") -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{slug}"


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


@app.command()
def main(
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained SAE directory. Default: latest under "
             "checkpoints/materials/sae/.",
    ),
    embeddings_dir: Path | None = typer.Option(
        None, "--embeddings-dir",
        help="Cached embeddings dir. Default: from SAE manifest.",
    ),
    out: Path = typer.Option(
        Path("runs/materials/sae_diagnostics"), "--out", "-o",
    ),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Run the SAE diagnostic and report health metrics."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.representation.sae import TopKSAE  # noqa: PLC0415

    if sae_dir is None:
        sae_dir = _latest_dir(Path("checkpoints/materials/sae"))
        if sae_dir is None:
            raise typer.BadParameter(
                "no SAE under checkpoints/materials/sae/."
            )
    sae_path = sae_dir / "sae.pt"
    payload = torch.load(sae_path, map_location=device, weights_only=False)

    if embeddings_dir is None:
        recorded = payload.get("embeddings_dir")
        if recorded and Path(recorded).exists():
            embeddings_dir = Path(recorded)
        else:
            embeddings_dir = _latest_dir(Path("runs/materials/embeddings"))
    if embeddings_dir is None:
        raise typer.BadParameter("could not resolve embeddings dir")
    emb_path = embeddings_dir / "embeddings.npy"
    if not emb_path.exists():
        raise typer.BadParameter(f"missing {emb_path}")

    embeddings = np.load(emb_path).astype(np.float32)
    n = int(embeddings.shape[0])

    sae = TopKSAE(
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        k=int(payload["k"]),
    ).to(device)
    sae.load_state_dict(payload["state_dict"], strict=True)
    sae.eval()
    cls_mean = torch.from_numpy(
        np.asarray(payload["cls_mean"], dtype=np.float32).reshape(-1)
    ).to(device)
    cls_std = torch.from_numpy(
        np.asarray(payload["cls_std"], dtype=np.float32).reshape(-1)
    ).to(device)

    run_id = _generate_run_id()
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 6b: SAE diagnostic")
    typer.echo(f"    sae_dir         : {sae_dir}")
    typer.echo(f"    embeddings_dir  : {embeddings_dir}")
    typer.echo(f"    n_specimens     : {n}")
    typer.echo(
        f"    SAE config      : in_dim={sae.in_dim} "
        f"hidden_dim={sae.hidden_dim} k={sae.k}"
    )
    typer.echo("")

    # Forward all embeddings through the SAE.
    BATCH = 4096
    activations_blocks: list[np.ndarray] = []
    recon_err_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, n, BATCH):
            x = torch.from_numpy(embeddings[s : s + BATCH]).to(device)
            x_norm = (x - cls_mean) / cls_std.clamp_min(1.0e-6)
            z = sae.encode(x_norm)
            recon = sae.decode(z)
            err = ((x_norm - recon) ** 2).mean(dim=-1)
            activations_blocks.append(z.detach().cpu().numpy())
            recon_err_blocks.append(err.detach().cpu().numpy())
    activations = np.concatenate(activations_blocks, axis=0)
    recon_err = np.concatenate(recon_err_blocks, axis=0)

    # ----- Per-feature activation counts ---------------------------------
    fired = activations > 1.0e-6                     # (N, hidden_dim)
    activation_counts = fired.sum(axis=0)            # (hidden_dim,)
    n_dead = int((activation_counts == 0).sum())
    n_rare = int(((activation_counts > 0) & (activation_counts < 10)).sum())
    n_alive = int((activation_counts >= 10).sum())

    activation_count_quantiles = {
        "min": int(activation_counts.min()),
        "p01": int(np.quantile(activation_counts, 0.01)),
        "p10": int(np.quantile(activation_counts, 0.10)),
        "p50": int(np.quantile(activation_counts, 0.50)),
        "p90": int(np.quantile(activation_counts, 0.90)),
        "p99": int(np.quantile(activation_counts, 0.99)),
        "max": int(activation_counts.max()),
        "mean": float(activation_counts.mean()),
    }

    # ----- Per-specimen feature-set diversity ----------------------------
    unique_features_used = int((activation_counts > 0).sum())
    coverage_frac = unique_features_used / sae.hidden_dim

    # Mean overlap between the top-k masks of pairs of specimens
    # (sample 256 pairs to keep this cheap)
    rng = np.random.default_rng(0)
    n_pairs = min(256, n * (n - 1) // 2)
    if n_pairs > 0:
        idx_a = rng.integers(0, n, size=n_pairs)
        idx_b = rng.integers(0, n, size=n_pairs)
        same = idx_a == idx_b
        idx_b[same] = (idx_b[same] + 1) % n
        overlap = (
            (fired[idx_a] & fired[idx_b]).sum(axis=-1) / max(sae.k, 1)
        ).astype(np.float32)
        mean_pair_overlap = float(overlap.mean())
    else:
        mean_pair_overlap = float("nan")

    # ----- Reconstruction error distribution -----------------------------
    recon_err_quantiles = {
        "min": float(recon_err.min()),
        "p10": float(np.quantile(recon_err, 0.10)),
        "p50": float(np.quantile(recon_err, 0.50)),
        "p90": float(np.quantile(recon_err, 0.90)),
        "p99": float(np.quantile(recon_err, 0.99)),
        "max": float(recon_err.max()),
        "mean": float(recon_err.mean()),
    }

    # ----- Reporting -----------------------------------------------------
    typer.echo("=========================================================")
    typer.echo("SAE diagnostic")
    typer.echo("=========================================================")
    typer.echo(f"  hidden_dim                 : {sae.hidden_dim}")
    typer.echo(f"  k (Top-K constraint)       : {sae.k}")
    typer.echo(f"  n_specimens                : {n}")
    typer.echo("")
    typer.echo("Feature usage:")
    typer.echo(
        f"  dead features (0 fires)    : {n_dead:>5} / {sae.hidden_dim} "
        f"({100.0 * n_dead / sae.hidden_dim:.1f}%)"
    )
    typer.echo(
        f"  rare features (<10 fires)  : {n_rare:>5} / {sae.hidden_dim} "
        f"({100.0 * n_rare / sae.hidden_dim:.1f}%)"
    )
    typer.echo(
        f"  alive features (>=10 fires): {n_alive:>5} / {sae.hidden_dim} "
        f"({100.0 * n_alive / sae.hidden_dim:.1f}%)"
    )
    typer.echo(
        f"  total unique features used : {unique_features_used} "
        f"(coverage {100.0 * coverage_frac:.1f}%)"
    )
    typer.echo("")
    typer.echo("Per-feature activation count distribution:")
    for k, v in activation_count_quantiles.items():
        typer.echo(f"  {k:>5}: {v}")
    typer.echo("")
    typer.echo(
        f"Mean pairwise overlap between top-k masks  : "
        f"{mean_pair_overlap:.4f}"
    )
    typer.echo(
        f"  (uniform-random expectation = k/hidden_dim = "
        f"{sae.k / sae.hidden_dim:.4f})"
    )
    typer.echo("")
    typer.echo("Reconstruction error per-specimen distribution (normalized space):")
    for k, v in recon_err_quantiles.items():
        typer.echo(f"  {k:>5}: {v:.6f}")
    typer.echo("")

    # ----- Verdict -------------------------------------------------------
    issues = []
    if n_dead / sae.hidden_dim > 0.20:
        issues.append(
            f"high dead-feature rate ({100.0 * n_dead / sae.hidden_dim:.1f}%); "
            "consider dead-feature resampling (Bricken et al. 2023) or "
            "JumpReLU SAE (Lieberum et al. 2024)"
        )
    if n_rare / sae.hidden_dim > 0.50:
        issues.append(
            f"many rare features ({100.0 * n_rare / sae.hidden_dim:.1f}%); "
            "may indicate the embedding distribution is concentrated "
            "and a smaller hidden_dim would suffice"
        )
    if mean_pair_overlap > 5.0 * (sae.k / sae.hidden_dim):
        issues.append(
            "high pairwise overlap between top-k masks; specimens are "
            "activating very similar feature sets, which suggests mode "
            "collapse or insufficient hidden_dim"
        )
    if recon_err_quantiles["p99"] > 5.0 * recon_err_quantiles["p50"]:
        issues.append(
            "long-tailed reconstruction error; some specimens are poorly "
            "covered by the feature dictionary; consider larger hidden_dim "
            "or longer training"
        )

    if not issues:
        verdict = (
            "HEALTHY: dead-feature rate, activation distribution, mask "
            "overlap, and reconstruction error all in expected ranges. "
            "No SAE upgrade needed before Stage 7."
        )
    else:
        verdict = "ISSUES FOUND:\n  - " + "\n  - ".join(issues)
    typer.echo(verdict)

    # ----- Persist YAML --------------------------------------------------
    report = {
        "run_id": run_id,
        "completed_utc": datetime.now(UTC).isoformat(),
        "sae_dir": str(sae_dir),
        "embeddings_dir": str(embeddings_dir),
        "config": {
            "in_dim": int(sae.in_dim),
            "hidden_dim": int(sae.hidden_dim),
            "k": int(sae.k),
            "n_specimens": int(n),
        },
        "feature_usage": {
            "n_dead": n_dead,
            "n_rare": n_rare,
            "n_alive": n_alive,
            "n_unique_used": unique_features_used,
            "coverage_fraction": coverage_frac,
        },
        "activation_count_quantiles": activation_count_quantiles,
        "mean_pair_overlap": mean_pair_overlap,
        "uniform_pair_overlap_expectation": sae.k / sae.hidden_dim,
        "recon_err_quantiles": recon_err_quantiles,
        "verdict": verdict,
    }
    with (out_dir / "diagnostic.yaml").open("w") as f:
        yaml.safe_dump(report, f, sort_keys=False)
    typer.echo("")
    typer.echo(f"==> Report: {out_dir / 'diagnostic.yaml'}")


if __name__ == "__main__":
    app()
