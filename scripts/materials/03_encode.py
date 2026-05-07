"""Stage 3: forward CHGNet over all train specimens, cache pooled embeddings.

Amortizes the CHGNet forward across the downstream stages (probes,
SAE, labelling, CoT-record build, single-shot inference) so they
each load embeddings from disk instead of re-running CHGNet.

Output:

    runs/materials/embeddings/<run_id>/embeddings.npy        # (N, fea_dim) fp32
    runs/materials/embeddings/<run_id>/specimen_ids.npy      # (N,) int32
    runs/materials/embeddings/<run_id>/manifest.yaml

Each row of ``embeddings.npy`` corresponds to the specimen id at
the same row of ``specimen_ids.npy``. Failed forwards (oversized
cells, malformed structures) are skipped; their ids are NOT in
``specimen_ids.npy``, which serves as the canonical "encoded
subset" downstream stages train on.

Default subset: train split from ``splits.yaml`` (i.e. all
specimens except the held-out 200). Use ``--include-holdout`` to
include the held-out specimens as well; this is needed by stage 9
(single-shot inference) but not by training stages.

Usage:

    bash scripts/materials/03_encode.sh

Depends on:
    typer, h5py, torch, chgnet, pymatgen.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _generate_run_id(slug: str = "encode") -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{slug}"


def _load_split_ids(splits_path: Path, split_name: str) -> list[int]:
    with splits_path.open("r") as f:
        splits = yaml.safe_load(f)
    if split_name in splits:
        return [int(x) for x in splits[split_name]]
    raise typer.BadParameter(f"unknown split {split_name!r}")


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    splits_path: Path = typer.Option(
        Path("data/materials_project_v1/splits.yaml"), "--splits-path",
    ),
    out: Path = typer.Option(
        Path("runs/materials/embeddings"), "--out", "-o",
    ),
    include_holdout: bool = typer.Option(
        False, "--include-holdout/--no-include-holdout",
    ),
    chgnet_model_name: str = typer.Option(
        "0.3.0", "--chgnet-model-name",
    ),
    max_atoms: int = typer.Option(80, "--max-atoms"),
    n_max: int = typer.Option(
        0, "--n-max",
        help="Cap on number of specimens to encode (0 = no cap).",
    ),
    log_every: int = typer.Option(500, "--log-every"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Forward CHGNet over the requested split, cache pooled embeddings."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not h5_path.exists():
        typer.echo(f"ERROR: {h5_path} not found.", err=True)
        sys.exit(2)
    if not splits_path.exists():
        typer.echo(f"ERROR: {splits_path} not found.", err=True)
        sys.exit(2)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.chgnet_wrap import (  # noqa: PLC0415
        CHGNetWrap, structure_from_arrays,
    )

    train_ids = _load_split_ids(splits_path, "train")
    holdout_ids = _load_split_ids(splits_path, "holdout") if include_holdout else []
    target_ids = list(train_ids) + list(holdout_ids)
    if n_max > 0:
        target_ids = target_ids[:n_max]

    run_id = _generate_run_id()
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 3: encode")
    typer.echo(f"    h5_path          : {h5_path}")
    typer.echo(f"    splits_path      : {splits_path}")
    typer.echo(f"    include_holdout  : {include_holdout}")
    typer.echo(f"    n target         : {len(target_ids)}")
    typer.echo(f"    chgnet           : {chgnet_model_name}")
    typer.echo(f"    device           : {device}")
    typer.echo(f"    out_dir          : {out_dir}")
    typer.echo("")

    typer.echo("==> Loading CHGNet...")
    wrap = CHGNetWrap.load(device=device, model_name=chgnet_model_name)

    embeddings_blocks: list[np.ndarray] = []
    kept_ids: list[int] = []
    n_skipped_size = 0
    n_skipped_error = 0
    fea_dim: int | None = None
    t0 = time.time()

    with h5py.File(h5_path, "r") as h5:
        element_names_attr = h5.attrs.get("element_names")
        element_names = (
            [s.decode() if isinstance(s, bytes) else str(s)
             for s in element_names_attr]
            if element_names_attr is not None else []
        )
        for i, sid in enumerate(target_ids):
            n_atoms = int(np.asarray(h5["nsites"][sid]))
            if n_atoms > max_atoms or n_atoms < 1:
                n_skipped_size += 1
                continue
            species_ids = np.asarray(h5["n_atoms_padded"][sid])[:n_atoms]
            positions = np.asarray(h5["positions_padded"][sid])[:n_atoms]
            lattice = np.asarray(h5["lattice"][sid])
            try:
                structure = structure_from_arrays(
                    species_ids=species_ids,
                    positions=positions,
                    lattice=lattice,
                    element_names=element_names,
                )
                _, pooled = wrap.encode(structure)
            except Exception as exc:
                if n_skipped_error < 5:
                    typer.echo(f"    skip sid={sid}: {exc!r}")
                n_skipped_error += 1
                continue
            pooled_np = pooled.detach().cpu().numpy().astype(np.float32)
            if fea_dim is None:
                fea_dim = int(pooled_np.shape[0])
            embeddings_blocks.append(pooled_np)
            kept_ids.append(int(sid))
            if (i + 1) % log_every == 0 or i + 1 == len(target_ids):
                typer.echo(
                    f"    {i + 1:>6}/{len(target_ids)} "
                    f"kept={len(kept_ids)} "
                    f"skipped_size={n_skipped_size} "
                    f"skipped_error={n_skipped_error} "
                    f"elapsed={time.time() - t0:.1f}s"
                )

    if not embeddings_blocks:
        typer.echo("ERROR: no embeddings produced.", err=True)
        sys.exit(3)

    embeddings = np.stack(embeddings_blocks, axis=0)
    specimen_ids = np.asarray(kept_ids, dtype=np.int32)

    typer.echo("")
    typer.echo(f"==> Saving embeddings: shape={embeddings.shape}")
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "specimen_ids.npy", specimen_ids)

    manifest = {
        "run_id": run_id,
        "completed_utc": datetime.now(UTC).isoformat(),
        "h5_path": str(h5_path),
        "splits_path": str(splits_path),
        "include_holdout": include_holdout,
        "chgnet_model_name": chgnet_model_name,
        "max_atoms": max_atoms,
        "n_target": len(target_ids),
        "n_kept": int(specimen_ids.shape[0]),
        "n_skipped_size": int(n_skipped_size),
        "n_skipped_error": int(n_skipped_error),
        "fea_dim": int(fea_dim or 0),
        "wall_clock_seconds": float(time.time() - t0),
    }
    with (out_dir / "manifest.yaml").open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    typer.echo(f"==> Manifest: {out_dir / 'manifest.yaml'}")


if __name__ == "__main__":
    app()
