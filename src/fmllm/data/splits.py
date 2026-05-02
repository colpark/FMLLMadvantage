"""Held-out split assignment for the synthetic Lennard-Jones dataset.

The generator produces 50,000 specimens. Phase 1 holds 10,000 of them
out for evaluation, partitioned along two axes:

    - Atom count N: in-distribution (``5, 7, 9, 11, 13``) versus
      out-of-distribution (``17, 19, 21, 25, 30``).
    - Temperature T: in-distribution (``T <= t_in_distribution_max``)
      versus out-of-distribution (``T > t_in_distribution_max``).

The held-out subset spans all four ``(N_axis, T_axis)`` cells. The
training subset gets the remaining specimens. Assignments live in a
YAML file so a reader can inspect them and downstream tools can pull
the same splits across runs.

Produces:
    Split assignments as a Python dict and a corresponding YAML file
    on disk.

Depends on:
    pyyaml, numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


CELL_LABELS = ("in_n_in_t", "in_n_ood_t", "ood_n_in_t", "ood_n_ood_t")
DEFAULT_TRAIN_SCALES: tuple[int, ...] = (10_000, 30_000, 50_000)
"""Default nested training scales for the FM-quality sweep (E5)."""


def cell_label(
    n_atoms: int,
    temperature: float,
    *,
    n_in_distribution: Sequence[int],
    t_in_distribution_max: float,
) -> str:
    """Return the ``(N_axis, T_axis)`` cell label for one specimen."""
    n_axis = "in_n" if n_atoms in set(n_in_distribution) else "ood_n"
    t_axis = "in_t" if temperature <= t_in_distribution_max else "ood_t"
    return f"{n_axis}_{t_axis}"


def assign_splits(
    *,
    specimen_ids: Sequence[int],
    atom_counts: Sequence[int],
    temperatures: Sequence[float],
    n_in_distribution: Sequence[int],
    t_in_distribution_max: float,
    num_holdout: int,
    seed: int = 0,
    nested_train_scales: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Assign each specimen to ``train`` or one of the four holdout cells.

    The function draws ``num_holdout`` specimen IDs uniformly at
    random, then labels each held-out specimen by its
    ``(N_axis, T_axis)`` cell. The remaining specimens form the full
    training pool. Nested subsets of the training pool serve the
    FM-quality sweep (E5): the function shuffles the pool with a
    derived seed and exposes prefix-sized subsets at the requested
    scales. ``train_X ⊆ train_Y`` whenever ``X <= Y``, and every
    nested subset is a subset of the full ``train`` list.

    Args:
        specimen_ids: Iterable of unique integer IDs.
        atom_counts: ``N`` per specimen, aligned with ``specimen_ids``.
        temperatures: ``T`` per specimen.
        n_in_distribution: The set of N values considered
            in-distribution.
        t_in_distribution_max: Upper bound of the in-distribution
            temperature interval (inclusive).
        num_holdout: Total number of held-out specimens.
        seed: Random seed for the holdout draw.
        nested_train_scales: Sizes of the nested training subsets used
            by the E5 quality sweep. Defaults to
            ``DEFAULT_TRAIN_SCALES`` (``10_000, 30_000, 50_000``). Each
            requested scale gets clamped to the available training
            pool size, so a scale larger than the pool resolves to the
            full pool.

    Returns:
        A dict with keys ``train`` (sorted list of IDs), ``holdout``
        (dict mapping cell label to sorted list of IDs),
        ``train_subsets`` (dict mapping scale label like ``train_10k``
        to a sorted list of IDs), and ``meta``.
    """
    ids = np.asarray(list(specimen_ids), dtype=np.int64)
    n_arr = np.asarray(list(atom_counts), dtype=np.int64)
    t_arr = np.asarray(list(temperatures), dtype=np.float64)

    if not (ids.shape[0] == n_arr.shape[0] == t_arr.shape[0]):
        raise ValueError(
            "specimen_ids, atom_counts, temperatures must have the same length"
        )
    if num_holdout < 0 or num_holdout > ids.shape[0]:
        raise ValueError(
            f"num_holdout must lie in [0, {ids.shape[0]}], got {num_holdout}"
        )

    rng = np.random.default_rng(seed)
    holdout_idx = rng.choice(ids.shape[0], size=num_holdout, replace=False)
    holdout_mask = np.zeros(ids.shape[0], dtype=bool)
    holdout_mask[holdout_idx] = True

    train_ids = sorted(int(x) for x in ids[~holdout_mask])

    holdout_buckets: dict[str, list[int]] = {label: [] for label in CELL_LABELS}
    n_in_set = set(int(x) for x in n_in_distribution)
    for i in holdout_idx:
        spec_id = int(ids[i])
        label = cell_label(
            int(n_arr[i]),
            float(t_arr[i]),
            n_in_distribution=n_in_set,
            t_in_distribution_max=t_in_distribution_max,
        )
        holdout_buckets[label].append(spec_id)
    for label in CELL_LABELS:
        holdout_buckets[label].sort()

    scales = (
        tuple(int(x) for x in nested_train_scales)
        if nested_train_scales is not None
        else DEFAULT_TRAIN_SCALES
    )
    if any(s <= 0 for s in scales):
        raise ValueError(f"nested_train_scales must be positive, got {scales}")

    rng_scale = np.random.default_rng(seed + 1000)
    shuffled = list(train_ids)
    rng_scale.shuffle(shuffled)
    train_subsets: dict[str, list[int]] = {}
    for scale in scales:
        actual = min(scale, len(shuffled))
        train_subsets[_scale_label(scale)] = sorted(int(x) for x in shuffled[:actual])

    return {
        "train": train_ids,
        "holdout": holdout_buckets,
        "train_subsets": train_subsets,
        "meta": {
            "num_specimens": int(ids.shape[0]),
            "num_holdout": int(num_holdout),
            "n_in_distribution": [int(x) for x in n_in_distribution],
            "t_in_distribution_max": float(t_in_distribution_max),
            "seed": int(seed),
            "nested_train_scales": list(scales),
            "nested_actual_sizes": {
                _scale_label(s): len(train_subsets[_scale_label(s)]) for s in scales
            },
        },
    }


def _scale_label(scale: int) -> str:
    """Return the canonical key for a training scale.

    Examples: 10_000 -> ``train_10k``, 30_000 -> ``train_30k``,
    50_000 -> ``train_50k``, 12_500 -> ``train_12500``.
    """
    if scale % 1000 == 0:
        return f"train_{scale // 1000}k"
    return f"train_{scale}"


def select_train_subset(splits: dict[str, Any], scale_label: str) -> list[int]:
    """Look up a nested training subset by its scale label.

    Falls back to the full ``train`` list if ``scale_label`` is
    ``"train_full"`` or matches the meta's largest scale.
    """
    if scale_label in {"train_full", "train"}:
        return list(splits.get("train", []))
    train_subsets = splits.get("train_subsets", {})
    if scale_label not in train_subsets:
        raise KeyError(
            f"unknown train scale {scale_label!r}; "
            f"available: {sorted(train_subsets.keys())}"
        )
    return list(train_subsets[scale_label])


def save_splits_yaml(splits: dict[str, Any], path: Path | str) -> Path:
    """Write splits to a YAML file. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(splits, f, sort_keys=False)
    return path


def load_splits_yaml(path: Path | str) -> dict[str, Any]:
    """Read splits from a YAML file written by :func:`save_splits_yaml`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"splits file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


__all__ = [
    "CELL_LABELS",
    "DEFAULT_TRAIN_SCALES",
    "assign_splits",
    "cell_label",
    "load_splits_yaml",
    "save_splits_yaml",
    "select_train_subset",
]
