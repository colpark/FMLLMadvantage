"""Stage 2: pack the per-material JSON cache into a single HDF5 file.

Reads ``data/materials_project_v1/raw/*.json.gz`` and writes
``data/materials_project_v1/specimens.h5`` with the schema documented
in ``docs/materials/data_pipeline.md``.

The HDF5 has fixed-shape arrays. Per-material atom counts vary, so
positions and species are stored padded to ``MAX_ATOMS`` with a
``padding_mask`` for the actual extent. Materials whose unit cell
exceeds ``MAX_ATOMS`` are skipped with a logged count.

Usage:

    bash scripts/materials/01_build_mp_h5.sh

Depends on:
    typer, h5py, numpy.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import typer


app = typer.Typer(add_completion=False, no_args_is_help=False)


CRYSTAL_SYSTEMS = (
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic",
)


def _crystal_system_id(name: str | None) -> int:
    if not name:
        return -1
    name_l = str(name).strip().lower()
    for i, cs in enumerate(CRYSTAL_SYSTEMS):
        if cs == name_l:
            return i
    return -1


def _record_paths(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*.json.gz"))


def _load_record(path: Path) -> dict | None:
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception:
        return None


def _structure_to_arrays(struct: dict, element_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (species_ids, positions, lattice) for one material's structure dict.

    The ``structure`` dict is whatever pymatgen Structure.as_dict()
    produces; we extract Cartesian coordinates and the lattice matrix.
    """
    sites = struct.get("sites") or []
    n_atoms = len(sites)
    species_ids = np.full(n_atoms, -1, dtype=np.int8)
    positions = np.zeros((n_atoms, 3), dtype=np.float32)
    for i, site in enumerate(sites):
        species_list = site.get("species") or [{}]
        # First-occupancy element (most MP entries are stoichiometric).
        elem = species_list[0].get("element") or species_list[0].get("specie") or ""
        species_ids[i] = element_to_idx.get(str(elem), -1)
        # Prefer Cartesian xyz; fall back to abc + lattice if needed.
        xyz = site.get("xyz")
        if xyz is None:
            abc = site.get("abc") or [0.0, 0.0, 0.0]
            xyz = abc
        positions[i] = [float(c) for c in xyz[:3]]
    lattice = struct.get("lattice") or {}
    matrix = lattice.get("matrix") or [[0.0]*3]*3
    lattice_arr = np.asarray(matrix, dtype=np.float32)
    return species_ids, positions, lattice_arr


@app.command()
def main(
    raw_dir: Path = typer.Option(
        Path("data/materials_project_v1/raw"), "--raw-dir",
    ),
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    max_atoms: int = typer.Option(
        80, "--max-atoms",
        help="Truncate per-cell atom count. 80 covers ~99% of MP "
             "structures with e_above_hull < 0.5 eV/atom.",
    ),
    min_atoms: int = typer.Option(1, "--min-atoms"),
    progress_every: int = typer.Option(2000, "--progress-every"),
) -> None:
    """Pack the raw JSON cache into HDF5."""
    paths = _record_paths(raw_dir)
    if not paths:
        typer.echo(
            f"ERROR: no .json.gz under {raw_dir}. Run stage 1 first.",
            err=True,
        )
        sys.exit(2)
    typer.echo(f"==> Packing {len(paths)} materials -> {h5_path}")
    typer.echo(f"    max_atoms = {max_atoms}, min_atoms = {min_atoms}")

    # First pass: collect element vocabulary.
    typer.echo("==> First pass: building element vocabulary")
    element_set: set[str] = set()
    for i, p in enumerate(paths):
        rec = _load_record(p)
        if rec is None:
            continue
        for el in rec.get("elements") or []:
            element_set.add(str(el))
        if (i + 1) % progress_every == 0:
            typer.echo(f"    {i + 1:>6}/{len(paths)} elements_seen={len(element_set)}")
    element_names = sorted(element_set)
    element_to_idx = {e: i for i, e in enumerate(element_names)}
    typer.echo(f"    element vocabulary size = {len(element_names)}")

    # Second pass: collect records that fit max_atoms, accumulate arrays.
    typer.echo("==> Second pass: filtering + packing arrays")
    keep_records: list[dict] = []
    n_dropped_too_big = 0
    n_dropped_too_small = 0
    n_dropped_malformed = 0
    for i, p in enumerate(paths):
        rec = _load_record(p)
        if rec is None:
            n_dropped_malformed += 1
            continue
        nsites = int(rec.get("nsites", 0) or 0)
        if nsites > max_atoms:
            n_dropped_too_big += 1
            continue
        if nsites < min_atoms:
            n_dropped_too_small += 1
            continue
        keep_records.append(rec)
        if (i + 1) % progress_every == 0:
            typer.echo(
                f"    {i + 1:>6}/{len(paths)} kept={len(keep_records)} "
                f"too_big={n_dropped_too_big} too_small={n_dropped_too_small} "
                f"malformed={n_dropped_malformed}"
            )

    if not keep_records:
        typer.echo("ERROR: no valid records after filtering.", err=True)
        sys.exit(3)

    n = len(keep_records)
    typer.echo(f"==> Writing HDF5 with {n} specimens")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if h5_path.exists():
        h5_path.unlink()

    # Pre-allocate arrays.
    material_ids = np.array([r["material_id"] for r in keep_records], dtype="S20")
    formulas = np.array([r["formula_pretty"] for r in keep_records], dtype="S40")
    nsites_arr = np.array([r["nsites"] for r in keep_records], dtype=np.int32)
    volume_arr = np.array([r["volume"] for r in keep_records], dtype=np.float32)
    density_arr = np.array([r["density"] for r in keep_records], dtype=np.float32)
    e_form_arr = np.array(
        [r["formation_energy_per_atom"] for r in keep_records], dtype=np.float32,
    )
    e_hull_arr = np.array(
        [r["energy_above_hull"] for r in keep_records], dtype=np.float32,
    )
    bg_arr = np.array([r["band_gap"] for r in keep_records], dtype=np.float32)
    is_metal_arr = np.array([bool(r["is_metal"]) for r in keep_records], dtype=bool)
    mag_arr = np.array(
        [(r.get("total_magnetization") if r.get("total_magnetization") is not None else float("nan")) for r in keep_records],
        dtype=np.float32,
    )
    sg_num_arr = np.array(
        [int((r.get("symmetry") or {}).get("space_group_number") or 1) for r in keep_records],
        dtype=np.int32,
    )
    cs_id_arr = np.array(
        [_crystal_system_id((r.get("symmetry") or {}).get("crystal_system")) for r in keep_records],
        dtype=np.int32,
    )

    species_padded = np.full((n, max_atoms), -1, dtype=np.int8)
    positions_padded = np.zeros((n, max_atoms, 3), dtype=np.float32)
    padding_mask = np.zeros((n, max_atoms), dtype=bool)
    lattice_arr = np.zeros((n, 3, 3), dtype=np.float32)

    typer.echo("    converting structures ...")
    for i, rec in enumerate(keep_records):
        struct = rec.get("structure") or {}
        sids, pos, lat = _structure_to_arrays(struct, element_to_idx)
        nat = sids.shape[0]
        species_padded[i, :nat] = sids
        positions_padded[i, :nat] = pos
        padding_mask[i, :nat] = True
        lattice_arr[i] = lat
        if (i + 1) % progress_every == 0:
            typer.echo(f"    {i + 1:>6}/{n} structures packed")

    typer.echo("    writing HDF5 ...")
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["n_specimens"] = n
        h5.attrs["element_names"] = np.array(element_names, dtype="S6")
        h5.attrs["max_atoms"] = max_atoms
        h5.attrs["created_utc"] = datetime.now(UTC).isoformat()
        h5.attrs["crystal_systems"] = np.array(CRYSTAL_SYSTEMS, dtype="S20")

        h5.create_dataset("material_id", data=material_ids)
        h5.create_dataset("formula_pretty", data=formulas)
        h5.create_dataset("nsites", data=nsites_arr)
        h5.create_dataset("volume", data=volume_arr)
        h5.create_dataset("density", data=density_arr)
        h5.create_dataset("formation_energy_per_atom", data=e_form_arr)
        h5.create_dataset("energy_above_hull", data=e_hull_arr)
        h5.create_dataset("band_gap", data=bg_arr)
        h5.create_dataset("is_metal", data=is_metal_arr)
        h5.create_dataset("total_magnetization", data=mag_arr)
        h5.create_dataset("space_group_number", data=sg_num_arr)
        h5.create_dataset("crystal_system_id", data=cs_id_arr)
        h5.create_dataset("n_atoms_padded", data=species_padded)
        h5.create_dataset("positions_padded", data=positions_padded)
        h5.create_dataset("lattice", data=lattice_arr)
        h5.create_dataset("padding_mask", data=padding_mask)

    typer.echo("==> Done.")
    typer.echo(f"    n_specimens   : {n}")
    typer.echo(f"    too_big       : {n_dropped_too_big}")
    typer.echo(f"    too_small     : {n_dropped_too_small}")
    typer.echo(f"    malformed     : {n_dropped_malformed}")
    typer.echo(f"    elements      : {len(element_names)}")
    typer.echo(f"    output        : {h5_path}")


if __name__ == "__main__":
    app()
