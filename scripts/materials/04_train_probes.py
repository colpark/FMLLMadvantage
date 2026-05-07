"""Stage 4: train a probe bank on the cached CHGNet embeddings.

Each probe is a 1- or 2-layer MLP on the (fea_dim)-dim pooled
embedding. Targets:

    formation_energy (regression)
    e_above_hull     (regression)
    band_gap         (regression)
    is_metal         (classification, 2-class)
    space_group      (classification, top-K most-common labels)

Output:

    checkpoints/materials/probes/<run_id>/
        probe_*.pt
        manifest.yaml

The probe-bank module (``fmllm.training.probe_bank``) is reused
unchanged from the LJ pipeline; only the targets change.

Usage:

    bash scripts/materials/04_train_probes.sh

Depends on:
    typer, h5py, numpy, torch, pyyaml.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml
from torch import nn


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _generate_run_id(slug: str = "probes") -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{slug}"


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _load_targets(
    h5: h5py.File, specimen_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pull per-specimen targets from the materials HDF5."""
    return {
        "formation_energy": np.asarray(
            h5["formation_energy_per_atom"][:][specimen_ids], dtype=np.float32,
        ),
        "e_above_hull": np.asarray(
            h5["energy_above_hull"][:][specimen_ids], dtype=np.float32,
        ),
        "band_gap": np.asarray(
            h5["band_gap"][:][specimen_ids], dtype=np.float32,
        ),
        "is_metal": np.asarray(
            h5["is_metal"][:][specimen_ids], dtype=np.int64,
        ),
        "space_group": np.asarray(
            h5["space_group_number"][:][specimen_ids], dtype=np.int64,
        ),
    }


def _train_regression(
    *, name: str, X: torch.Tensor, y: torch.Tensor,
    in_dim: int, hidden: int, epochs: int, lr: float,
    batch_size: int, device: str,
) -> tuple[nn.Module, dict]:
    """Train a regression probe; return (module, target_stats)."""
    from fmllm.training.probe_bank import _build_head  # noqa: PLC0415

    mod = _build_head(in_dim=in_dim, out_dim=1, hidden=hidden).to(device)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr)
    n = X.shape[0]
    target_mean = float(y.mean().item())
    target_std = float(y.std().item())
    target_min = float(y.min().item())
    target_max = float(y.max().item())
    for epoch in range(epochs):
        perm = torch.randperm(n)
        loss_sum = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb, yb = X[idx], y[idx]
            pred = mod(xb).squeeze(-1)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            nb += 1
        if epoch == 0 or epoch == epochs - 1:
            typer.echo(
                f"    [{name}] epoch {epoch:>2}/{epochs} "
                f"mse={loss_sum / max(nb, 1):.5f}"
            )
    return mod, {
        "target_mean": target_mean,
        "target_std": target_std,
        "target_min": target_min,
        "target_max": target_max,
    }


def _train_classification(
    *, name: str, X: torch.Tensor, y: torch.Tensor, n_classes: int,
    in_dim: int, hidden: int, epochs: int, lr: float,
    batch_size: int, device: str, class_names: list[str],
) -> tuple[nn.Module, dict]:
    from fmllm.training.probe_bank import _build_head  # noqa: PLC0415

    mod = _build_head(in_dim=in_dim, out_dim=n_classes, hidden=hidden).to(device)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        loss_sum = 0.0
        correct = 0
        total = 0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb, yb = X[idx], y[idx]
            logits = mod(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            total += int(yb.shape[0])
            nb += 1
        if epoch == 0 or epoch == epochs - 1:
            typer.echo(
                f"    [{name}] epoch {epoch:>2}/{epochs} "
                f"loss={loss_sum / max(nb, 1):.5f} acc={correct / max(total, 1):.4f}"
            )
    return mod, {"class_names": class_names}


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    embeddings_dir: Path | None = typer.Option(
        None, "--embeddings-dir",
        help="Directory with embeddings.npy + specimen_ids.npy. "
             "Default: latest under runs/materials/embeddings/.",
    ),
    out: Path = typer.Option(
        Path("checkpoints/materials/probes"), "--out", "-o",
    ),
    hidden: int = typer.Option(128, "--hidden"),
    epochs: int = typer.Option(30, "--epochs"),
    lr: float = typer.Option(1.0e-3, "--lr"),
    batch_size: int = typer.Option(256, "--batch-size"),
    space_group_top_k: int = typer.Option(
        20, "--space-group-top-k",
        help="Train the space-group probe on the top-K most-common "
             "space groups; specimens outside the top-K are dropped "
             "during training. K=20 covers ~75-80% of MP structures.",
    ),
    device: str = typer.Option("auto", "--device"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Train the materials probe bank on cached embeddings."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.training.probe_bank import ProbeBank, ProbeSpec  # noqa: PLC0415

    if embeddings_dir is None:
        cand = _latest_dir(Path("runs/materials/embeddings"))
        if cand is None:
            raise typer.BadParameter(
                "no embeddings under runs/materials/embeddings/. Run "
                "scripts/materials/03_encode.sh first."
            )
        embeddings_dir = cand

    emb_path = embeddings_dir / "embeddings.npy"
    sid_path = embeddings_dir / "specimen_ids.npy"
    if not emb_path.exists() or not sid_path.exists():
        raise typer.BadParameter(f"missing embeddings under {embeddings_dir}")
    embeddings = np.load(emb_path).astype(np.float32)
    specimen_ids = np.load(sid_path).astype(np.int64)
    in_dim = int(embeddings.shape[1])

    run_id = _generate_run_id()
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 4: train probes")
    typer.echo(f"    embeddings_dir : {embeddings_dir}")
    typer.echo(f"    n_specimens    : {len(specimen_ids)}")
    typer.echo(f"    in_dim         : {in_dim}")
    typer.echo(f"    hidden         : {hidden}")
    typer.echo(f"    epochs         : {epochs}")
    typer.echo(f"    out_dir        : {out_dir}")
    typer.echo("")

    with h5py.File(h5_path, "r") as h5:
        targets = _load_targets(h5, specimen_ids)

    X = torch.from_numpy(embeddings).to(device)

    bank = ProbeBank()

    # Regression probes -----------------------------------------------------
    for name in ("formation_energy", "e_above_hull", "band_gap"):
        y = torch.from_numpy(targets[name].astype(np.float32)).to(device)
        mod, stats = _train_regression(
            name=name, X=X, y=y, in_dim=in_dim, hidden=hidden,
            epochs=epochs, lr=lr, batch_size=batch_size, device=device,
        )
        spec = ProbeSpec(
            name=name, kind="regression", in_dim=in_dim, out_dim=1,
            hidden=hidden,
            target_min=stats["target_min"], target_max=stats["target_max"],
            target_mean=stats["target_mean"], target_std=stats["target_std"],
        )
        bank.add(spec, mod)

    # Classification probes ------------------------------------------------
    # is_metal: binary
    y_metal = torch.from_numpy(targets["is_metal"]).to(device)
    mod, _ = _train_classification(
        name="is_metal", X=X, y=y_metal, n_classes=2, in_dim=in_dim,
        hidden=hidden, epochs=epochs, lr=lr, batch_size=batch_size,
        device=device, class_names=["non_metal", "metal"],
    )
    bank.add(
        ProbeSpec(
            name="is_metal", kind="classification", in_dim=in_dim,
            out_dim=2, hidden=hidden,
            class_names=["non_metal", "metal"],
        ),
        mod,
    )

    # space_group: top-K most-common -- map to compact label space, drop rest
    sg = targets["space_group"].astype(np.int64)
    bins = np.bincount(sg, minlength=231)
    top_k_groups = list(np.argsort(-bins)[:space_group_top_k])
    sg_lookup = {int(g): i for i, g in enumerate(top_k_groups)}
    in_top = np.array([int(g) in sg_lookup for g in sg.tolist()])
    if int(in_top.sum()) > 0:
        X_sg = torch.from_numpy(embeddings[in_top]).to(device)
        y_sg = torch.from_numpy(
            np.array([sg_lookup[int(g)] for g in sg[in_top].tolist()],
                     dtype=np.int64),
        ).to(device)
        class_names = [f"sg{int(g)}" for g in top_k_groups]
        mod, _ = _train_classification(
            name="space_group", X=X_sg, y=y_sg, n_classes=len(top_k_groups),
            in_dim=in_dim, hidden=hidden, epochs=epochs, lr=lr,
            batch_size=batch_size, device=device, class_names=class_names,
        )
        bank.add(
            ProbeSpec(
                name="space_group", kind="classification", in_dim=in_dim,
                out_dim=len(top_k_groups), hidden=hidden,
                class_names=class_names,
            ),
            mod,
        )
    else:
        typer.echo("    [space_group] no specimens in top-K; skipping.")

    bank.save(out_dir)
    typer.echo("")
    typer.echo(f"==> Probes saved at {out_dir}")
    typer.echo(f"    probes : {list(bank.specs.keys())}")

    extra_manifest = {
        "embeddings_dir": str(embeddings_dir),
        "n_specimens": int(len(specimen_ids)),
        "in_dim": in_dim,
        "hidden": hidden,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "space_group_top_k": space_group_top_k,
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    with (out_dir / "extras.yaml").open("w") as f:
        yaml.safe_dump(extra_manifest, f, sort_keys=False)


if __name__ == "__main__":
    app()
