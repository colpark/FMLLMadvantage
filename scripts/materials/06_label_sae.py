"""Stage 6: label SAE features by correlation with materials attributes.

Mirrors ``scripts/label_sae_features.py`` from the LJ pipeline. For
every feature in the trained SAE, find the top activating specimens
and describe them via:

    crystal_system     : categorical lock (cubic / hexagonal / ...)
    is_metal           : categorical lock (True / False)
    band_gap_class     : categorical lock (metal / narrow / wide)
    formation_energy   : continuous correlation
    e_above_hull       : continuous correlation
    band_gap           : continuous correlation
    n_atoms            : continuous correlation

Output:

    runs/materials/sae_labels/<run_id>/labels.json
    runs/materials/sae_labels/<run_id>/details.yaml
    runs/materials/sae_labels/<run_id>/manifest.yaml

Usage:

    bash scripts/materials/06_label_sae.sh

Depends on:
    typer, h5py, numpy, torch, pyyaml.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _generate_run_id(slug: str = "sae-labels") -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{slug}"


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _load_sae(
    sae_path: Path, device: str,
) -> tuple[object, np.ndarray, np.ndarray]:
    from fmllm.representation.sae import TopKSAE  # noqa: PLC0415

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


def _band_gap_class(bg: float, is_metal: bool) -> str:
    if is_metal or bg <= 1.0e-3:
        return "metal"
    if bg <= 3.0:
        return "narrow"
    return "wide"


def _crystal_system_name(idx: int, names: list[str]) -> str:
    if 0 <= idx < len(names):
        return names[idx]
    return "?"


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained SAE directory. Default: latest under "
             "checkpoints/materials/sae/.",
    ),
    embeddings_dir: Path | None = typer.Option(
        None, "--embeddings-dir",
        help="Cached CHGNet embeddings dir. Default: the one used by "
             "the SAE training run if recorded; otherwise latest.",
    ),
    out: Path = typer.Option(
        Path("runs/materials/sae_labels"), "--out", "-o",
    ),
    top_n: int = typer.Option(50, "--top-n"),
    min_purity: float = typer.Option(
        0.90, "--min-purity",
        help="v2 default: 0.90 (was 0.70). Sharper categorical locks.",
    ),
    min_corr: float = typer.Option(
        0.55, "--min-corr",
        help="v2 default: 0.55 (was 0.30). Only strong correlations "
             "produce continuous tags.",
    ),
    corr_on_top_n: bool = typer.Option(
        True, "--corr-on-top-n/--corr-on-all",
        help="v2 default: compute correlations on top-N activators "
             "instead of all specimens. Sharper signal.",
    ),
    top_specimens_keep: int = typer.Option(
        5, "--top-specimens-keep",
        help="How many representative training specimens to attach "
             "per feature (v2 grounding for the CoT generator).",
    ),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Label every SAE feature using materials attribute correlations."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.labels import label_materials_feature  # noqa: PLC0415

    if sae_dir is None:
        sae_dir = _latest_dir(Path("checkpoints/materials/sae"))
        if sae_dir is None:
            raise typer.BadParameter(
                "no SAE under checkpoints/materials/sae/. Run "
                "scripts/materials/05_train_sae.sh first."
            )
    sae_path = sae_dir / "sae.pt"
    if not sae_path.exists():
        raise typer.BadParameter(f"missing {sae_path}")

    if embeddings_dir is None:
        try:
            payload = torch.load(sae_path, map_location="cpu", weights_only=False)
            recorded = payload.get("embeddings_dir")
            if recorded and Path(recorded).exists():
                embeddings_dir = Path(recorded)
        except Exception:
            embeddings_dir = None
    if embeddings_dir is None:
        embeddings_dir = _latest_dir(Path("runs/materials/embeddings"))
        if embeddings_dir is None:
            raise typer.BadParameter(
                "no embeddings under runs/materials/embeddings/."
            )

    emb_path = embeddings_dir / "embeddings.npy"
    sid_path = embeddings_dir / "specimen_ids.npy"
    if not emb_path.exists() or not sid_path.exists():
        raise typer.BadParameter(f"missing embeddings under {embeddings_dir}")
    embeddings = np.load(emb_path).astype(np.float32)
    specimen_ids = np.load(sid_path).astype(np.int64)

    run_id = _generate_run_id()
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 6: label SAE features")
    typer.echo(f"    sae_dir              : {sae_dir}")
    typer.echo(f"    embeddings_dir       : {embeddings_dir}")
    typer.echo(f"    n_specimens          : {len(specimen_ids)}")
    typer.echo(f"    top_n                : {top_n}")
    typer.echo(f"    min_purity           : {min_purity}")
    typer.echo(f"    min_corr             : {min_corr}")
    typer.echo(f"    corr_on_top_n        : {corr_on_top_n}")
    typer.echo(f"    top_specimens_keep   : {top_specimens_keep}")
    typer.echo(f"    out_dir              : {out_dir}")
    typer.echo("")

    sae, cls_mean, cls_std = _load_sae(sae_path, device=device)
    typer.echo(
        f"    SAE config: in_dim={sae.in_dim} "
        f"hidden_dim={sae.hidden_dim} k={sae.k}"
    )

    # Forward all embeddings through the SAE encoder.
    cls_mean_t = torch.from_numpy(cls_mean.reshape(-1)).to(device)
    cls_std_t = torch.from_numpy(cls_std.reshape(-1)).to(device)
    activations: list[np.ndarray] = []
    BATCH = 4096
    with torch.no_grad():
        for s in range(0, len(specimen_ids), BATCH):
            x = torch.from_numpy(embeddings[s : s + BATCH]).to(device)
            x_norm = (x - cls_mean_t) / cls_std_t.clamp_min(1.0e-6)
            z = sae.encode(x_norm)
            activations.append(z.detach().cpu().numpy())
    activations_arr = np.concatenate(activations, axis=0)
    typer.echo(
        f"    activations shape: {activations_arr.shape}, "
        f"mean active fraction: {(activations_arr > 0).mean():.4f}"
    )

    # Pull per-specimen attributes.
    typer.echo("==> Pulling materials attributes...")
    with h5py.File(h5_path, "r") as h5:
        crystal_system_names_attr = h5.attrs.get("crystal_systems")
        if crystal_system_names_attr is not None:
            cs_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in crystal_system_names_attr
            ]
        else:
            cs_names = [
                "triclinic", "monoclinic", "orthorhombic",
                "tetragonal", "trigonal", "hexagonal", "cubic",
            ]
        cs_ids = h5["crystal_system_id"][:][specimen_ids]
        is_metal_arr = np.asarray(h5["is_metal"][:][specimen_ids], dtype=bool)
        e_form_arr = np.asarray(
            h5["formation_energy_per_atom"][:][specimen_ids], dtype=np.float32,
        )
        e_hull_arr = np.asarray(
            h5["energy_above_hull"][:][specimen_ids], dtype=np.float32,
        )
        bg_arr = np.asarray(
            h5["band_gap"][:][specimen_ids], dtype=np.float32,
        )
        n_atoms_arr = np.asarray(
            h5["nsites"][:][specimen_ids], dtype=np.float32,
        )
        # v2: pull material_id and formula for top-specimen grounding.
        material_ids: list[str] = []
        formulas: list[str] = []
        if "material_id" in h5:
            mid_raw = h5["material_id"][:][specimen_ids]
            material_ids = [
                m.decode() if isinstance(m, bytes) else str(m)
                for m in mid_raw
            ]
        if "formula_pretty" in h5:
            fmt_raw = h5["formula_pretty"][:][specimen_ids]
            formulas = [
                f.decode() if isinstance(f, bytes) else str(f)
                for f in fmt_raw
            ]
    cs_strings = np.array(
        [_crystal_system_name(int(i), cs_names) for i in cs_ids], dtype=object,
    )
    bg_class_strings = np.array(
        [_band_gap_class(float(bg), bool(im))
         for bg, im in zip(bg_arr, is_metal_arr, strict=True)],
        dtype=object,
    )

    # Label every feature.
    typer.echo("==> Labelling features...")
    labels_str: dict[int, str] = {}
    labels_rich: dict[str, dict] = {}
    details: list[dict] = []
    n_locked = 0
    n_unlabelled = 0
    n_rare = 0
    for i in range(activations_arr.shape[1]):
        feat = activations_arr[:, i]
        rec = label_materials_feature(
            feature_idx=i,
            feature_activations=feat,
            crystal_systems=cs_strings,
            is_metals=is_metal_arr,
            band_gap_classes=bg_class_strings,
            formation_energies=e_form_arr,
            e_above_hulls=e_hull_arr,
            band_gaps=bg_arr,
            n_atoms=n_atoms_arr,
            top_n=top_n,
            min_purity=min_purity,
            min_corr=min_corr,
            material_ids=material_ids if material_ids else None,
            formulas=formulas if formulas else None,
            corr_on_top_n=corr_on_top_n,
            top_specimens_keep=top_specimens_keep,
        )
        labels_str[i] = rec.label
        # v2 rich record (separate file so v1 readers still work).
        labels_rich[str(i)] = {
            "label": rec.label,
            "label_rich": rec.label_rich,
            "tags": list(rec.tags),
            "top_specimens": list(rec.top_specimens),
            "activation_quantiles": rec.activation_quantiles,
            "n_top_activators": rec.n_top_activators,
            "purities": {
                "crystal_system": rec.crystal_system_purity,
                "is_metal": rec.is_metal_purity,
                "band_gap_class": rec.band_gap_class_purity,
            },
            "correlations": {
                "formation_energy": rec.formation_energy_corr,
                "e_above_hull": rec.e_above_hull_corr,
                "band_gap": rec.band_gap_corr,
                "n_atoms": rec.n_atoms_corr,
            },
        }
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

    labels_path = out_dir / "labels.json"
    with labels_path.open("w") as f:
        json.dump(labels_str, f, indent=2)

    labels_rich_path = out_dir / "labels_rich.json"
    with labels_rich_path.open("w") as f:
        json.dump(labels_rich, f, indent=2)

    details_path = out_dir / "details.yaml"
    with details_path.open("w") as f:
        yaml.safe_dump({"features": details}, f, sort_keys=False)

    with (out_dir / "manifest.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "sae_dir": str(sae_dir),
                "embeddings_dir": str(embeddings_dir),
                "h5_path": str(h5_path),
                "n_specimens": int(len(specimen_ids)),
                "top_n": top_n,
                "min_purity": min_purity,
                "min_corr": min_corr,
                "corr_on_top_n": corr_on_top_n,
                "top_specimens_keep": top_specimens_keep,
                "n_features": int(activations_arr.shape[1]),
                "n_locked": n_locked,
                "n_unlabelled": n_unlabelled,
                "n_rare": n_rare,
                "labels_path": str(labels_path),
                "labels_rich_path": str(labels_rich_path),
                "completed_utc": datetime.now(UTC).isoformat(),
            },
            f,
            sort_keys=False,
        )

    typer.echo(f"==> Labels (v1): {labels_path}")
    typer.echo(f"==> Labels (v2 rich): {labels_rich_path}")


if __name__ == "__main__":
    app()
