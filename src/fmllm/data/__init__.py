"""Synthetic Lennard-Jones dataset generation and loading.

Public modules:
    splits: held-out partitioning logic.
    dataset: HDF5-backed PyTorch Dataset.
    generator: CLI script that produces the full dataset.

The module re-exports the most common helpers so callers can write
``from fmllm.data import LJSpecimenDataset`` without remembering which
file holds which symbol.
"""

from fmllm.data.dataset import LJSpecimenDataset
from fmllm.data.splits import (
    CELL_LABELS,
    assign_splits,
    cell_label,
    load_splits_yaml,
    save_splits_yaml,
)

__all__ = [
    "CELL_LABELS",
    "LJSpecimenDataset",
    "assign_splits",
    "cell_label",
    "load_splits_yaml",
    "save_splits_yaml",
]
