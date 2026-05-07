"""HDF5 reader for Materials Project specimens.

Mirrors the LJ ``LJSpecimenDataset`` interface for the materials
port. Lazy-loads the HDF5 file (so processes can fork cleanly) and
provides random access to (positions, lattice, species, properties)
per material.

Depends on:
    h5py, numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class MaterialsSpecimen:
    """One material's content as numpy arrays.

    All arrays are unpadded; the padding mask is applied during
    reading so callers can rely on the array lengths matching
    ``n_atoms``.
    """

    material_id: str
    formula_pretty: str
    n_atoms: int
    species_ids: np.ndarray            # (n_atoms,) int8 -- index into element vocab
    positions: np.ndarray              # (n_atoms, 3) float32 -- Cartesian Å
    lattice: np.ndarray                # (3, 3) float32 -- Å
    volume: float
    density: float
    formation_energy_per_atom: float
    energy_above_hull: float
    band_gap: float
    is_metal: bool
    total_magnetization: float         # NaN -> 0.0 in this representation
    space_group_number: int
    crystal_system_id: int


class MaterialsSpecimenDataset:
    """Random-access HDF5 reader for the materials specimens file.

    Usage::

        dataset = MaterialsSpecimenDataset("data/materials_project_v1/specimens.h5")
        spec = dataset[42]
        print(spec.material_id, spec.formation_energy_per_atom, spec.n_atoms)

    The file handle is opened lazily on first access so the dataset
    object is safe to share across DataLoader workers / multi-process
    setups.
    """

    def __init__(self, h5_path: str | Path) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        self._h5: h5py.File | None = None
        self._element_names: list[str] | None = None
        self._n: int | None = None

    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            attrs = dict(self._h5.attrs)
            self._n = int(attrs.get("n_specimens", len(self._h5["material_id"])))
            raw = attrs.get("element_names")
            if raw is not None:
                self._element_names = [
                    s.decode() if isinstance(s, bytes) else str(s) for s in raw
                ]
        return self._h5

    @property
    def element_names(self) -> list[str]:
        self._ensure_open()
        return list(self._element_names or [])

    def __len__(self) -> int:
        self._ensure_open()
        return int(self._n or 0)

    def __getitem__(self, idx: int) -> MaterialsSpecimen:
        h5 = self._ensure_open()
        n_atoms = int(np.asarray(h5["nsites"][idx]))
        # Padded arrays use -1 for padding sites.
        species_padded = np.asarray(h5["n_atoms_padded"][idx])  # (MAX_ATOMS,)
        positions_padded = np.asarray(h5["positions_padded"][idx])  # (MAX_ATOMS, 3)
        species = species_padded[:n_atoms].astype(np.int8)
        positions = positions_padded[:n_atoms].astype(np.float32)
        lattice = np.asarray(h5["lattice"][idx]).astype(np.float32)
        mid = h5["material_id"][idx]
        if isinstance(mid, bytes):
            mid = mid.decode()
        formula = h5["formula_pretty"][idx]
        if isinstance(formula, bytes):
            formula = formula.decode()
        mag = float(np.asarray(h5["total_magnetization"][idx]))
        if np.isnan(mag):
            mag = 0.0
        return MaterialsSpecimen(
            material_id=str(mid),
            formula_pretty=str(formula),
            n_atoms=n_atoms,
            species_ids=species,
            positions=positions,
            lattice=lattice,
            volume=float(np.asarray(h5["volume"][idx])),
            density=float(np.asarray(h5["density"][idx])),
            formation_energy_per_atom=float(
                np.asarray(h5["formation_energy_per_atom"][idx])
            ),
            energy_above_hull=float(np.asarray(h5["energy_above_hull"][idx])),
            band_gap=float(np.asarray(h5["band_gap"][idx])),
            is_metal=bool(np.asarray(h5["is_metal"][idx])),
            total_magnetization=mag,
            space_group_number=int(np.asarray(h5["space_group_number"][idx])),
            crystal_system_id=int(np.asarray(h5["crystal_system_id"][idx])),
        )

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


__all__ = ["MaterialsSpecimen", "MaterialsSpecimenDataset"]
