"""Tests for the cross-FM tolerance utility."""

from __future__ import annotations

from pathlib import Path

import pytest

from fmllm.fms._calibration import (
    CrossFMToleranceMatrix,
    compute_cross_fm_tolerances,
    load_tolerance_matrix,
    save_tolerance_matrix,
)


def _records(n: int = 100) -> list[dict[str, dict[str, float]]]:
    """Synthetic records: FM1 and FM2 disagree slightly on energy."""
    import random
    rng = random.Random(0)
    return [
        {
            "energy_per_atom": {
                "fm1_image": -1.0 + rng.gauss(0.0, 0.1),
                "fm2_rdf": -1.0 + rng.gauss(0.0, 0.1),
            },
            "atom_count": {
                "fm1_image": 7,
                "fm3_traj": 7,
            },
        }
        for _ in range(n)
    ]


def test_compute_tolerances_returns_pair_summaries():
    matrix = compute_cross_fm_tolerances(records=_records(200))
    pairs = {(p.variable, p.fm_a, p.fm_b) for p in matrix.pairwise}
    assert ("energy_per_atom", "fm1_image", "fm2_rdf") in pairs
    assert ("atom_count", "fm1_image", "fm3_traj") in pairs


def test_thresholds_increase_with_alpha():
    """The 90% threshold (alpha=0.10) sits above the 80% threshold (alpha=0.20)."""
    matrix = compute_cross_fm_tolerances(records=_records(500))
    for pair in matrix.pairwise:
        if pair.variable != "energy_per_atom":
            continue
        t_10 = pair.thresholds["alpha_0.1000"]
        t_20 = pair.thresholds["alpha_0.2000"]
        assert t_10 >= t_20


def test_tolerance_matrix_yaml_round_trip(tmp_path: Path):
    matrix = compute_cross_fm_tolerances(records=_records(50))
    path = tmp_path / "cross_fm.yaml"
    save_tolerance_matrix(matrix, path)
    loaded = load_tolerance_matrix(path)
    assert loaded.model_dump() == matrix.model_dump()


def test_compute_tolerances_handles_empty_records():
    matrix = compute_cross_fm_tolerances(records=[])
    assert isinstance(matrix, CrossFMToleranceMatrix)
    assert matrix.pairwise == []


def test_compute_tolerances_skips_singleton_sources():
    """Variables present in only one FM produce no pairwise entry."""
    records = [
        {"only_one": {"fm1_image": 0.5}},
        {"only_one": {"fm1_image": 0.7}},
    ]
    matrix = compute_cross_fm_tolerances(records=records)
    assert matrix.pairwise == []
