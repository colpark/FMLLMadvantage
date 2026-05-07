"""Ground-truth extractor for Materials Project specimens.

Mirrors the LJ ``_truth_dict`` helper in shape: a thin function that
reads from the materials HDF5 and returns the structured ground
truth dict the CoT generator and probe trainer consume.

The canonical materials ground-truth schema:

    formation_energy : float    eV/atom (DFT)
    e_above_hull     : float    eV/atom (DFT, distance to convex hull)
    is_stable        : bool     e_above_hull <= STABILITY_THRESHOLD
    band_gap         : float    eV (DFT)
    band_gap_class   : str      one of {metal, narrow, wide}
    space_group      : int      1..230
    crystal_system   : str      one of {triclinic, ..., cubic}
    is_metal         : bool
    total_magnetization : float | None    μB
    n_atoms          : int      number of atoms in the unit cell

Depends on:
    h5py, numpy.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


# Conventions for derived classes ------------------------------------------

STABILITY_THRESHOLD = 0.025      # eV/atom; the standard MP cutoff for stable
NARROW_GAP_MAX = 3.0             # eV; "narrow" gap is 0 < band_gap <= this
                                 # "metal" is band_gap = 0; "wide" is > NARROW_GAP_MAX

CRYSTAL_SYSTEMS = (
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic",
)


def band_gap_class(band_gap: float, is_metal: bool | None = None) -> str:
    """Classify a DFT band gap as metal / narrow / wide.

    The ``is_metal`` boolean from MP is preferred when available,
    otherwise we infer from the band-gap value.
    """
    if is_metal is True:
        return "metal"
    if band_gap <= 1.0e-3:
        return "metal"
    if band_gap <= NARROW_GAP_MAX:
        return "narrow"
    return "wide"


def crystal_system_name(idx: int) -> str:
    """Return the crystal-system name for a 0..6 index."""
    if 0 <= idx < len(CRYSTAL_SYSTEMS):
        return CRYSTAL_SYSTEMS[idx]
    return "?"


def truth_dict(h5: h5py.File, sid: int) -> dict[str, Any]:
    """Extract the materials ground-truth dict for one specimen index.

    The HDF5 layout is the one produced by
    ``scripts/materials/01_build_mp_h5.py``.
    """
    e_form = float(np.asarray(h5["formation_energy_per_atom"][sid]))
    e_hull = float(np.asarray(h5["energy_above_hull"][sid]))
    bg = float(np.asarray(h5["band_gap"][sid]))
    is_metal_arr = h5["is_metal"][sid] if "is_metal" in h5 else None
    is_metal = bool(np.asarray(is_metal_arr)) if is_metal_arr is not None else (bg <= 1.0e-3)
    sg = int(np.asarray(h5["space_group_number"][sid]))
    cs_idx = int(np.asarray(h5["crystal_system_id"][sid])) if "crystal_system_id" in h5 else -1
    n_atoms = int(np.asarray(h5["nsites"][sid]))
    mag_arr = h5["total_magnetization"][sid] if "total_magnetization" in h5 else None
    mag = float(np.asarray(mag_arr)) if mag_arr is not None else 0.0
    if np.isnan(mag):
        mag = 0.0
    return {
        "formation_energy": e_form,
        "e_above_hull": e_hull,
        "is_stable": bool(e_hull <= STABILITY_THRESHOLD),
        "band_gap": bg,
        "band_gap_class": band_gap_class(bg, is_metal=is_metal),
        "space_group": sg,
        "crystal_system": crystal_system_name(cs_idx),
        "is_metal": is_metal,
        "total_magnetization": mag,
        "n_atoms": n_atoms,
    }


def is_correct(claim: dict, gt: dict) -> bool:
    """Strict joint correctness for a materials commit.

    Mirrors the LJ correctness criterion:
      * formation_energy within 0.05 eV/atom
      * e_above_hull within 0.025 eV/atom
      * is_stable matches exactly
      * band_gap_class matches exactly (metal / narrow / wide)
      * space_group exact match

    Returns True iff all five components are correct.
    """
    if not claim:
        return False
    try:
        e_form_ok = abs(float(claim.get("formation_energy", -999.0)) - gt["formation_energy"]) <= 0.05
        e_hull_ok = abs(float(claim.get("e_above_hull", 999.0)) - gt["e_above_hull"]) <= 0.025
        stable_ok = bool(claim.get("is_stable", False)) == bool(gt["is_stable"])
        bg_ok = str(claim.get("band_gap_class", "")).lower() == gt["band_gap_class"].lower()
        sg_ok = int(claim.get("space_group", -1)) == gt["space_group"]
    except (TypeError, ValueError):
        return False
    return e_form_ok and e_hull_ok and stable_ok and bg_ok and sg_ok


__all__ = [
    "STABILITY_THRESHOLD",
    "NARROW_GAP_MAX",
    "CRYSTAL_SYSTEMS",
    "band_gap_class",
    "crystal_system_name",
    "is_correct",
    "truth_dict",
]
