"""Tests for the data module.

The tests cover the splits assignment logic, the splits YAML round trip,
and a tiny synthetic HDF5 file that exercises the
``LJSpecimenDataset`` reader. None of these require a GPU.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from fmllm.data.splits import (
    CELL_LABELS,
    assign_splits,
    cell_label,
    load_splits_yaml,
    save_splits_yaml,
    select_train_subset,
)


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def test_cell_label_in_n_in_t():
    label = cell_label(7, 0.5, n_in_distribution=[5, 7, 9, 11, 13], t_in_distribution_max=1.0)
    assert label == "in_n_in_t"


def test_cell_label_ood_n_ood_t():
    label = cell_label(25, 1.5, n_in_distribution=[5, 7, 9, 11, 13], t_in_distribution_max=1.0)
    assert label == "ood_n_ood_t"


def test_cell_label_in_n_ood_t():
    label = cell_label(13, 1.7, n_in_distribution=[5, 7, 9, 11, 13], t_in_distribution_max=1.0)
    assert label == "in_n_ood_t"


def test_cell_label_ood_n_in_t():
    label = cell_label(17, 0.5, n_in_distribution=[5, 7, 9, 11, 13], t_in_distribution_max=1.0)
    assert label == "ood_n_in_t"


def test_assign_splits_size_invariants():
    """Train + holdout sums to total. No specimen appears twice."""
    rng = np.random.default_rng(0)
    num = 200
    atom_counts = rng.choice([5, 7, 9, 13, 17, 25], size=num).tolist()
    temperatures = rng.uniform(0.1, 2.0, size=num).tolist()

    splits = assign_splits(
        specimen_ids=list(range(num)),
        atom_counts=atom_counts,
        temperatures=temperatures,
        n_in_distribution=[5, 7, 9, 11, 13],
        t_in_distribution_max=1.0,
        num_holdout=40,
        seed=42,
    )

    train = splits["train"]
    holdout = splits["holdout"]
    total_holdout = sum(len(v) for v in holdout.values())
    assert total_holdout == 40
    assert len(train) + total_holdout == num

    seen = set(train)
    for label in CELL_LABELS:
        for spec_id in holdout[label]:
            assert spec_id not in seen
            seen.add(spec_id)
    assert seen == set(range(num))


def test_assign_splits_deterministic():
    """Same seed reproduces the same split."""
    args = dict(
        specimen_ids=list(range(50)),
        atom_counts=[7] * 50,
        temperatures=[0.5] * 50,
        n_in_distribution=[5, 7, 9, 11, 13],
        t_in_distribution_max=1.0,
        num_holdout=10,
        seed=123,
    )
    a = assign_splits(**args)
    b = assign_splits(**args)
    assert a == b


def test_assign_splits_holdout_too_large():
    with pytest.raises(ValueError):
        assign_splits(
            specimen_ids=list(range(10)),
            atom_counts=[5] * 10,
            temperatures=[0.5] * 10,
            n_in_distribution=[5, 7, 9, 11, 13],
            t_in_distribution_max=1.0,
            num_holdout=20,
            seed=0,
        )


def test_nested_train_subsets_are_strictly_nested():
    """train_10k subset of train_30k subset of train_50k subset of train pool."""
    rng = np.random.default_rng(0)
    num = 200
    atom_counts = rng.choice([5, 7, 9, 13, 17, 25], size=num).tolist()
    temperatures = rng.uniform(0.1, 2.0, size=num).tolist()
    splits = assign_splits(
        specimen_ids=list(range(num)),
        atom_counts=atom_counts,
        temperatures=temperatures,
        n_in_distribution=[5, 7, 9, 11, 13],
        t_in_distribution_max=1.0,
        num_holdout=40,
        seed=42,
        nested_train_scales=[20, 60, 100],
    )
    train_pool = set(splits["train"])
    subsets = splits["train_subsets"]
    s20 = set(subsets["train_20"])
    s60 = set(subsets["train_60"])
    s100 = set(subsets["train_100"])
    assert s20 <= s60 <= s100 <= train_pool
    assert len(s20) == 20
    assert len(s60) == 60
    assert len(s100) == 100


def test_nested_scales_clamp_to_pool_size():
    """A nested scale larger than the train pool clamps to the pool size."""
    splits = assign_splits(
        specimen_ids=list(range(50)),
        atom_counts=[7] * 50,
        temperatures=[0.5] * 50,
        n_in_distribution=[5, 7, 9, 11, 13],
        t_in_distribution_max=1.0,
        num_holdout=10,
        seed=0,
        nested_train_scales=[10, 30, 100],
    )
    train_pool = splits["train"]
    assert len(splits["train_subsets"]["train_10"]) == 10
    assert len(splits["train_subsets"]["train_30"]) == 30
    # The 100 scale clamps to len(train_pool) = 40.
    assert len(splits["train_subsets"]["train_100"]) == len(train_pool)


def test_select_train_subset_returns_full_for_train_full():
    from fmllm.data.splits import select_train_subset
    splits = {"train": [1, 2, 3], "train_subsets": {"train_10k": [1]}}
    assert select_train_subset(splits, "train_full") == [1, 2, 3]
    assert select_train_subset(splits, "train_10k") == [1]
    with pytest.raises(KeyError):
        select_train_subset(splits, "train_99k")


def test_splits_yaml_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    num = 50
    atom_counts = rng.choice([5, 7, 17], size=num).tolist()
    temperatures = rng.uniform(0.1, 2.0, size=num).tolist()
    splits = assign_splits(
        specimen_ids=list(range(num)),
        atom_counts=atom_counts,
        temperatures=temperatures,
        n_in_distribution=[5, 7],
        t_in_distribution_max=1.0,
        num_holdout=10,
        seed=7,
    )
    path = tmp_path / "splits.yaml"
    save_splits_yaml(splits, path)
    loaded = load_splits_yaml(path)
    assert loaded == splits


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def _build_synthetic_h5(path: Path, num: int = 4, max_n: int = 7, n_steps: int = 5,
                       image_size: int = 8, rdf_bins: int = 16) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["max_n_atoms"] = max_n
        f.attrs["image_size"] = image_size
        f.attrs["rdf_bins"] = rdf_bins
        f.attrs["md_steps_per_specimen"] = n_steps
        f.create_dataset("images", data=rng.standard_normal((num, image_size, image_size)).astype("f4"))
        f.create_dataset("rdfs", data=rng.standard_normal((num, rdf_bins)).astype("f4"))
        f.create_dataset(
            "traj_positions",
            data=rng.standard_normal((num, n_steps + 1, max_n, 2)).astype("f4"),
        )
        f.create_dataset(
            "traj_velocities",
            data=rng.standard_normal((num, n_steps + 1, max_n, 2)).astype("f4"),
        )
        f.create_dataset(
            "equilibrium_positions",
            data=rng.standard_normal((num, max_n, 2)).astype("f4"),
        )
        f.create_dataset("atom_counts", data=np.array([3, 5, 7, 4], dtype="i4"))
        f.create_dataset("temperatures", data=np.linspace(0.2, 1.5, num).astype("f4"))
        f.create_dataset("motif_ids", data=np.zeros(num, dtype="i2"))
        f.create_dataset("seeds", data=np.arange(num, dtype="i8"))


def test_dataset_reads_synthetic_h5(tmp_path: Path):
    from fmllm.data.dataset import LJSpecimenDataset

    h5_path = tmp_path / "specimens.h5"
    _build_synthetic_h5(h5_path)

    ds = LJSpecimenDataset(h5_path)
    try:
        assert len(ds) == 4
        item = ds[0]
        assert item["image"].shape == (8, 8)
        assert item["rdf"].shape == (16,)
        assert item["traj_positions"].shape == (6, 7, 2)
        assert item["traj_velocities"].shape == (6, 7, 2)
        assert item["atom_count"] == 3
        assert item["specimen_id"] == 0
        assert item["atom_mask"].shape == (7,)
        assert int(item["atom_mask"].sum().item()) == 3
    finally:
        ds.close()


def test_specimen_id_is_the_hdf5_row(tmp_path: Path):
    """The specimen_id key returns the HDF5 row index, not the seed value."""
    from fmllm.data.dataset import LJSpecimenDataset

    h5_path = tmp_path / "specimens.h5"
    _build_synthetic_h5(h5_path)

    # _build_synthetic_h5 stores seeds = np.arange(num); replace with offsets
    # so seed != row to make the distinction visible.
    with h5py.File(h5_path, "a") as f:
        f["seeds"][...] = np.arange(4) + 100  # seeds are 100, 101, 102, 103

    ds = LJSpecimenDataset(h5_path, keys=("specimen_id", "seed", "atom_count"))
    try:
        # specimen_id == row index; seed is the per-specimen RNG seed.
        assert ds[0]["specimen_id"] == 0
        assert ds[0]["seed"] == 100
        assert ds[2]["specimen_id"] == 2
        assert ds[2]["seed"] == 102
    finally:
        ds.close()


def test_dataset_filter_by_specimen_ids(tmp_path: Path):
    from fmllm.data.dataset import LJSpecimenDataset

    h5_path = tmp_path / "specimens.h5"
    _build_synthetic_h5(h5_path)

    ds = LJSpecimenDataset(h5_path, specimen_ids=[2, 0])
    try:
        assert len(ds) == 2
        # The dataset preserves the order of the supplied IDs.
        assert ds[0]["atom_count"] == 7
        assert ds[1]["atom_count"] == 3
    finally:
        ds.close()
