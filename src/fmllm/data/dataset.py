"""HDF5-backed PyTorch dataset for the synthetic Lennard-Jones data.

The dataset reads a single HDF5 file produced by
:mod:`fmllm.data.generator` and yields one specimen per ``__getitem__``.
The HDF5 layout uses fixed-size arrays padded to ``max_n_atoms`` along
the atom axis, plus a per-specimen ``atom_count`` and a boolean
``atom_mask`` so consumers can ignore padded entries.

Datasets:
    images: ``(num_specimens, H, W)`` float32.
    rdfs: ``(num_specimens, rdf_bins)`` float32.
    traj_positions: ``(num_specimens, n_steps + 1, max_n, 2)`` float32.
    traj_velocities: ``(num_specimens, n_steps + 1, max_n, 2)`` float32.
    equilibrium_positions: ``(num_specimens, max_n, 2)`` float32.
    atom_counts: ``(num_specimens,)`` int32.
    temperatures: ``(num_specimens,)`` float32.
    motif_ids: ``(num_specimens,)`` int16.
    seeds: ``(num_specimens,)`` int64.

The per-item ``specimen_id`` returned by the dataset is the HDF5 row
index. The per-specimen RNG seed lives in the ``seeds`` HDF5 dataset
and the reader exposes it under the optional ``seed`` key.

Attributes (root group):
    max_n_atoms, image_size, image_pixel_size_lj, image_blur_radius_lj,
    rdf_bins, rdf_r_max, md_dt, md_steps_per_specimen, motif_names.

Produces:
    PyTorch Datasets that load specimens lazily from disk.

Depends on:
    h5py, numpy, torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LJSpecimenDataset(Dataset):
    """HDF5-backed dataset of Lennard-Jones specimens.

    The class supports filtering by an explicit list of specimen IDs,
    which is how the splits machinery selects a train or holdout view
    of the same underlying file.
    """

    def __init__(
        self,
        h5_path: Path | str,
        *,
        specimen_ids: Sequence[int] | None = None,
        keys: Iterable[str] | None = None,
    ) -> None:
        """Open the HDF5 file and prepare the index.

        Args:
            h5_path: Path to the HDF5 file produced by the generator.
            specimen_ids: Optional list of specimen IDs to expose. The
                IDs equal the row indices in the HDF5 datasets. ``None``
                exposes every specimen.
            keys: Optional iterable of dataset names to load per item.
                ``None`` loads the canonical set
                ``("image", "rdf", "traj_positions", "traj_velocities",
                "atom_count", "temperature", "atom_mask", "specimen_id")``.
        """
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self._h5: h5py.File | None = None
        self._open()
        assert self._h5 is not None

        total = int(self._h5["atom_counts"].shape[0])
        if specimen_ids is None:
            self._index = np.arange(total, dtype=np.int64)
        else:
            ids = np.asarray(list(specimen_ids), dtype=np.int64)
            if ids.size and (ids.min() < 0 or ids.max() >= total):
                raise IndexError(
                    f"specimen_ids contain out-of-range entries (total={total})"
                )
            self._index = ids

        default_keys = (
            "image", "rdf", "traj_positions", "traj_velocities",
            "atom_count", "temperature", "atom_mask", "specimen_id",
        )
        self._keys = tuple(keys) if keys is not None else default_keys
        self.max_n_atoms = int(self._h5.attrs.get("max_n_atoms", 0))

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | float]:
        if self._h5 is None:
            self._open()
        assert self._h5 is not None

        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"dataset index out of range: {idx}")
        row = int(self._index[idx])

        out: dict[str, Any] = {}
        for key in self._keys:
            if key == "image":
                out["image"] = torch.from_numpy(self._h5["images"][row].astype(np.float32))
            elif key == "rdf":
                out["rdf"] = torch.from_numpy(self._h5["rdfs"][row].astype(np.float32))
            elif key == "traj_positions":
                out["traj_positions"] = torch.from_numpy(
                    self._h5["traj_positions"][row].astype(np.float32)
                )
            elif key == "traj_velocities":
                out["traj_velocities"] = torch.from_numpy(
                    self._h5["traj_velocities"][row].astype(np.float32)
                )
            elif key == "equilibrium_positions":
                out["equilibrium_positions"] = torch.from_numpy(
                    self._h5["equilibrium_positions"][row].astype(np.float32)
                )
            elif key == "atom_count":
                out["atom_count"] = int(self._h5["atom_counts"][row])
            elif key == "temperature":
                out["temperature"] = float(self._h5["temperatures"][row])
            elif key == "motif_id":
                out["motif_id"] = int(self._h5["motif_ids"][row])
            elif key == "specimen_id":
                # The specimen ID is the HDF5 row index. The 'seeds'
                # dataset stores the per-specimen RNG seed used by the
                # generator, which is a separate concept exposed below.
                out["specimen_id"] = int(row)
            elif key == "seed":
                out["seed"] = int(self._h5["seeds"][row])
            elif key == "atom_mask":
                n = int(self._h5["atom_counts"][row])
                mask = np.zeros(self.max_n_atoms, dtype=bool)
                mask[:n] = True
                out["atom_mask"] = torch.from_numpy(mask)
            else:
                raise KeyError(f"unknown dataset key: {key}")
        return out

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["LJSpecimenDataset"]
