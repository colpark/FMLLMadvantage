"""Shared helpers for the evaluation tests.

The eight tests reuse a few primitives:

    - extract observations and final claims from saved trajectories,
    - compute a structural distance between two trajectories,
    - build an equivalence relation over specimens (by true ``(N, motif)``),
    - look up ground-truth values from the dataset HDF5,
    - decode an FM's bridged ``Prediction.value`` payload into the
      typed dataclass downstream code understands.

Depends on:
    h5py, numpy.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from fmllm.fms._schemas import BridgedFMOutput
from fmllm.orchestrator import StepType, Trajectory
from fmllm.verifier.schema import PhysicalStateClaim


# ---------------------------------------------------------------------------
# Trajectory inspection
# ---------------------------------------------------------------------------


def extract_observations(traj: Trajectory) -> dict[str, BridgedFMOutput]:
    """Return the most recent bridged output per FM in this trajectory."""
    out: dict[str, BridgedFMOutput] = {}
    for s in traj.steps:
        if s.step_type is StepType.OBSERVATION and s.bridged_output is not None:
            out[s.bridged_output.source.fm_name] = s.bridged_output
    return out


def extract_final_claim(traj: Trajectory) -> PhysicalStateClaim | None:
    """Return the trajectory's final committed claim, if any."""
    return traj.final_claim


def trajectory_action_signature(traj: Trajectory) -> tuple[str, ...]:
    """A coarse-grained string sequence summarizing actions taken."""
    parts: list[str] = []
    for s in traj.steps:
        if s.llm_action is None:
            continue
        if s.step_type is StepType.OBSERVATION and s.bridged_output is not None:
            parts.append(f"call_fm:{s.bridged_output.source.fm_name}")
        elif s.step_type is StepType.HYPOTHESIS:
            parts.append("hypothesize")
        elif s.step_type is StepType.FINAL:
            parts.append("commit")
    return tuple(parts)


def trajectory_outcome(traj: Trajectory) -> str:
    """Return a coarse string label for downstream grouping."""
    if traj.final_verdict is None:
        return "no_verdict"
    return traj.final_verdict.aggregate_decision.value


# ---------------------------------------------------------------------------
# Ground truth from dataset
# ---------------------------------------------------------------------------


def load_ground_truth(
    h5_path: Path | str,
    specimen_ids: Iterable[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Read per-specimen truth (N, T, motif name) from the HDF5 store.

    Returns a dict keyed by specimen ID.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        atom_counts = np.asarray(f["atom_counts"]).astype(np.int64)
        temperatures = np.asarray(f["temperatures"]).astype(np.float64)
        motif_ids = np.asarray(f["motif_ids"]).astype(np.int64)
        motif_names = []
        if "motif_names" in f.attrs:
            motif_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f.attrs["motif_names"]
            ]
        ids = list(specimen_ids) if specimen_ids is not None else list(range(atom_counts.shape[0]))
        out: dict[int, dict[str, Any]] = {}
        for sid in ids:
            sid_int = int(sid)
            mid = int(motif_ids[sid_int])
            motif = motif_names[mid] if mid < len(motif_names) else str(mid)
            out[sid_int] = {
                "n": int(atom_counts[sid_int]),
                "t": float(temperatures[sid_int]),
                "motif": motif,
            }
    return out


def physical_equivalence_class(truth: dict[str, Any]) -> tuple[int, str]:
    """Two specimens are physically equivalent when they share the same
    ``(N, motif)`` regardless of temperature."""
    return (int(truth["n"]), str(truth["motif"]))


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Levenshtein distance between two action signatures."""
    if not a and not b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,           # insertion
                prev[j] + 1,              # deletion
                prev[j - 1] + cost,       # substitution
            )
        prev = cur
    return prev[lb]


def claim_distance(
    a: PhysicalStateClaim | None,
    b: PhysicalStateClaim | None,
) -> float:
    """Distance between two typed claims.

    The metric weights atom-count disagreement heavily (the dominant
    structural fact), temperature on its absolute LJ scale, motif as
    a 0/1 indicator, and per-atom energy on its LJ scale.
    """
    if a is None or b is None:
        return float("inf")
    d = 0.0
    if a.n_atoms is not None and b.n_atoms is not None:
        d += abs(a.n_atoms - b.n_atoms)
    elif (a.n_atoms is None) != (b.n_atoms is None):
        d += 5.0
    if a.temperature is not None and b.temperature is not None:
        d += abs(a.temperature - b.temperature)
    if a.motif is not None and b.motif is not None:
        d += 0.0 if a.motif == b.motif else 1.0
    if (
        a.per_atom_potential_energy is not None
        and b.per_atom_potential_energy is not None
    ):
        d += abs(a.per_atom_potential_energy - b.per_atom_potential_energy)
    return d


__all__ = [
    "claim_distance",
    "edit_distance",
    "extract_final_claim",
    "extract_observations",
    "load_ground_truth",
    "physical_equivalence_class",
    "trajectory_action_signature",
    "trajectory_outcome",
]
