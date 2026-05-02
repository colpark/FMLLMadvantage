"""Pairwise empirical agreement tolerances on shared causal variables.

Each FM declares ``dependencies`` in its metadata, naming the upstream
or downstream variables the FM's prediction relates to. When two FMs
share a target variable, the cross-FM verifier source compares their
implied estimates of that variable on the same specimen. The tolerance
this module produces says: empirically, how much do their estimates
disagree on the calibration set, and what threshold should the
verifier use before flagging a mismatch.

For the year-1 testbed, the practical shared-variable pairs are:

    - FM1 atom_count_pred (count head) vs FM2 atom_count_implied
      (derived from energy scaling, not yet wired since FM2 does not
      emit a count). Hold this slot for Phase 4.
    - FM1 lj_energy_from_positions (computed by the bridge from FM1
      positions plus the LJ Hamiltonian) vs FM2 per_atom_potential_energy
      (FM2's direct output).
    - FM1 atom_count_pred vs FM3 trajectory_atom_count_implied
      (mask-derived count from the trajectory tensor; this is a
      sanity check rather than a learned cross-FM agreement).

The function below is generic. It accepts a list of per-specimen
records, where each record holds estimates of any number of named
variables from any number of named source FMs, and returns a matrix
of pairwise agreement tolerances at the requested miscoverage levels.

Produces:
    A :class:`CrossFMToleranceMatrix` saved as YAML the verifier reads.

Depends on:
    numpy, pyyaml, pydantic.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairwiseAgreement(_StrictModel):
    """Empirical agreement summary for one shared variable across two FMs."""

    variable: str
    fm_a: str
    fm_b: str
    n_specimens: int = Field(ge=0)
    median_abs_diff: float
    p90_abs_diff: float
    p95_abs_diff: float
    thresholds: dict[str, float] = Field(default_factory=dict)


class CrossFMToleranceMatrix(_StrictModel):
    """Top-level container for cross-FM agreement results."""

    train_split: str
    timestamp_utc: str
    pairwise: list[PairwiseAgreement] = Field(default_factory=list)


def compute_cross_fm_tolerances(
    *,
    records: list[dict[str, dict[str, float]]],
    alpha_levels: tuple[float, ...] = (0.10, 0.20),
    train_split: str = "train_50k",
) -> CrossFMToleranceMatrix:
    """Compute pairwise tolerances over a list of per-specimen records.

    Each record has the form
    ``{"variable_name": {"fm_a_name": value, "fm_b_name": value, ...}}``.
    The function pairs every FM-pair that emits the same variable and
    summarizes the absolute differences. It then computes split-conformal
    thresholds at each requested alpha level.
    """
    from fmllm.fms.common import split_conformal_quantile  # local import
    from fmllm.fms._schemas.probe_schema import now_utc_iso

    by_var_pair: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for rec in records:
        for variable, sources in rec.items():
            names = sorted(sources.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    val_a = sources.get(a)
                    val_b = sources.get(b)
                    if val_a is None or val_b is None:
                        continue
                    diff = abs(float(val_a) - float(val_b))
                    by_var_pair[(variable, a, b)].append(diff)

    pairwise: list[PairwiseAgreement] = []
    for (variable, a, b), diffs in by_var_pair.items():
        arr = np.asarray(diffs, dtype=np.float64)
        thresholds = {
            f"alpha_{level:.4f}": float(split_conformal_quantile(arr, level))
            for level in alpha_levels
        }
        pairwise.append(
            PairwiseAgreement(
                variable=variable,
                fm_a=a,
                fm_b=b,
                n_specimens=int(arr.size),
                median_abs_diff=float(np.median(arr)),
                p90_abs_diff=float(np.quantile(arr, 0.90)),
                p95_abs_diff=float(np.quantile(arr, 0.95)),
                thresholds=thresholds,
            )
        )

    return CrossFMToleranceMatrix(
        train_split=train_split,
        timestamp_utc=now_utc_iso(),
        pairwise=pairwise,
    )


def save_tolerance_matrix(
    matrix: CrossFMToleranceMatrix,
    path: Path | str,
) -> Path:
    """Write the tolerance matrix to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(matrix.model_dump(), f, sort_keys=False)
    return path


def load_tolerance_matrix(path: Path | str) -> CrossFMToleranceMatrix:
    """Read a tolerance matrix from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tolerance matrix not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return CrossFMToleranceMatrix.model_validate(raw)


__all__ = [
    "CrossFMToleranceMatrix",
    "PairwiseAgreement",
    "compute_cross_fm_tolerances",
    "load_tolerance_matrix",
    "save_tolerance_matrix",
]
