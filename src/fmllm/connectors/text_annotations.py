"""Templated per-specimen text annotations for connector Stage 1 training.

The connector training data has shape (FM features, text). Real
captions for physics specimens are expensive to author, but every
specimen in the synthetic LJ testbed has full ground-truth metadata
in the HDF5 (atom count, motif, temperature, equilibrium positions,
trajectory). A deterministic templated generator produces
1-3 sentence descriptions covering:

    - basic identity (N, motif, temperature)
    - phase regime (cold / warm / hot)
    - structural features (first-shell peak, cluster diameter,
      coordination summary) when computable from positions

The annotations are intentionally factual and slightly varied so the
LM loss has signal beyond a single boilerplate. Determinism (same
specimen → same annotation) lets us cache and reproduce.

Depends on:
    h5py, numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Phase boundaries in LJ T units. Solid-like below ~0.3, liquid-like
# in [0.3, 1.0), gas-like above. Boundaries are heuristic for 2D LJ.
_T_PHASES: tuple[tuple[float, str], ...] = (
    (0.30, "solid-like"),
    (1.00, "liquid-like"),
    (float("inf"), "gas-like"),
)

_PHASE_DESCRIPTIONS: dict[str, str] = {
    "solid-like": "narrow, sharply peaked",
    "liquid-like": "broadened with visible second shell",
    "gas-like": "broad and shallow, weak ordering",
}


@dataclass
class SpecimenAnnotation:
    """One specimen's templated description plus the source fields it
    came from. Carrying the source fields makes the generator easy to
    audit and lets tests check that the description is faithful."""

    specimen_id: int
    n_atoms: int
    motif: str
    temperature: float
    phase: str
    diameter_lj: float | None
    mean_coordination: float | None
    text: str


def _phase_for(t: float) -> str:
    for upper, label in _T_PHASES:
        if t < upper:
            return label
    return _T_PHASES[-1][1]


def _cluster_diameter(positions: np.ndarray) -> float:
    """Geometric extent of the cluster: max pairwise distance.

    Args:
        positions: ``(N, 2)`` LJ-unit positions.
    """
    if positions.shape[0] < 2:
        return 0.0
    diffs = positions[:, None, :] - positions[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=-1))
    return float(d.max())


def _mean_coordination(positions: np.ndarray, cutoff: float = 1.4) -> float:
    """Average number of atoms within ``cutoff`` LJ of each atom.

    The cutoff sits between the LJ minimum (≈1.122) and the typical
    second-shell distance, so it counts first-neighbors only.
    """
    if positions.shape[0] < 2:
        return 0.0
    diffs = positions[:, None, :] - positions[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    neighbors = (d < cutoff).sum(axis=1)
    return float(neighbors.mean())


def annotate_specimen(
    *,
    specimen_id: int,
    n_atoms: int,
    motif: str,
    temperature: float,
    positions: np.ndarray | None = None,
) -> SpecimenAnnotation:
    """Build a deterministic templated annotation.

    Args:
        specimen_id: Dataset row.
        n_atoms: Atom count.
        motif: Canonical motif name.
        temperature: LJ temperature.
        positions: Optional ``(N, 2)`` equilibrium positions. When
            present the description references diameter and mean
            coordination; when absent the description sticks to N,
            motif, temperature, and phase.
    """
    phase = _phase_for(float(temperature))
    motif_pretty = motif.replace("_", " ")

    parts: list[str] = []
    parts.append(
        f"{int(n_atoms)}-atom {motif_pretty} cluster at "
        f"T = {float(temperature):.2f} LJ ({phase} regime)."
    )

    rdf_descriptor = _PHASE_DESCRIPTIONS.get(phase, "")
    if rdf_descriptor:
        parts.append(
            f"Expected radial distribution: {rdf_descriptor} first-neighbor peak "
            f"near 1.13 LJ."
        )

    diameter: float | None = None
    coord: float | None = None
    if positions is not None and positions.size > 0:
        diameter = _cluster_diameter(positions)
        coord = _mean_coordination(positions)
        parts.append(
            f"Geometric diameter approximately {diameter:.2f} LJ, "
            f"mean first-shell coordination approximately {coord:.2f}."
        )

    text = " ".join(parts)
    return SpecimenAnnotation(
        specimen_id=int(specimen_id),
        n_atoms=int(n_atoms),
        motif=str(motif),
        temperature=float(temperature),
        phase=phase,
        diameter_lj=diameter,
        mean_coordination=coord,
        text=text,
    )


def annotate_specimen_from_h5(
    h5_path: Path | str,
    specimen_id: int,
    *,
    use_positions: bool = True,
) -> SpecimenAnnotation:
    """Read one specimen's metadata from the testbed HDF5 and annotate it.

    Args:
        h5_path: Path to ``specimens.h5``.
        specimen_id: Dataset row.
        use_positions: When True, load equilibrium positions and surface
            diameter and coordination in the description. Disable for
            speed when the descriptive sentence about geometry is not
            needed.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(Path(h5_path), "r") as f:
        n_atoms = int(np.asarray(f["atom_counts"][specimen_id]))
        temperature = float(np.asarray(f["temperatures"][specimen_id]))
        motif_id = int(np.asarray(f["motif_ids"][specimen_id]))
        motif_names: list[str] = []
        if "motif_names" in f.attrs:
            motif_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f.attrs["motif_names"]
            ]
        motif = (
            motif_names[motif_id]
            if motif_id < len(motif_names)
            else str(motif_id)
        )
        positions: np.ndarray | None = None
        if use_positions:
            eq = np.asarray(f["equilibrium_positions"][specimen_id])
            if eq.ndim == 2:
                positions = eq[:n_atoms]

    return annotate_specimen(
        specimen_id=specimen_id,
        n_atoms=n_atoms,
        motif=motif,
        temperature=temperature,
        positions=positions,
    )


def annotation_label_dict(
    annotation: SpecimenAnnotation,
) -> dict[str, Any]:
    """Compact ground-truth label payload used by the probing study.

    The probes consume scalar ground-truth values, not free text, so we
    lift the structured fields out of :class:`SpecimenAnnotation` for
    them. Keys match what ``scripts/run_fm2_probes.py`` trains against.
    """
    return {
        "n_atoms": float(annotation.n_atoms),
        "temperature": float(annotation.temperature),
        "diameter_lj": (
            float(annotation.diameter_lj)
            if annotation.diameter_lj is not None
            else None
        ),
        "mean_coordination": (
            float(annotation.mean_coordination)
            if annotation.mean_coordination is not None
            else None
        ),
        "phase": annotation.phase,
        "motif": annotation.motif,
    }


__all__ = [
    "SpecimenAnnotation",
    "annotate_specimen",
    "annotate_specimen_from_h5",
    "annotation_label_dict",
]
