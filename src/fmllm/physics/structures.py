"""Generators for canonical 2D cluster motifs.

This module returns equilibrium positions for 2D atomic clusters at
zero temperature, intended as starting configurations for MD
equilibration in the dataset generator. All distances use reduced LJ
units. The triangular-lattice motifs use the LJ minimum spacing
``r_min = 2 ** (1/6)`` as the lattice constant, which places nearest
neighbors at the LJ pair-energy minimum.

The module exposes one canonical motif per N along with a small set of
alternates for variety in the synthetic dataset:

    - ``linear``: a straight chain centered at the origin.
    - ``ring``: a regular polygon at fixed nearest-neighbor distance.
    - ``triangular_disk``: the ``N`` atoms closest to the origin on a
      2D triangular (hexagonal) lattice. Cluster sizes ``N = 7`` and
      ``N = 13`` produce the closed-shell cluster shapes that play the
      role of the 3D pentagonal-bipyramid and icosahedron motifs in
      this 2D testbed.

Produces:
    Tensors of shape ``(N, 2)`` with float32 positions.

Depends on:
    torch.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from fmllm.physics.lj_potential import R_MIN

DEFAULT_SPACING = R_MIN
"""Nearest-neighbor distance used by the structure generators."""


# Mapping from N to a tuple of valid motif names. The first entry is
# the canonical default for that N. The generator picks one at random.
VALID_MOTIFS_FOR_N: dict[int, tuple[str, ...]] = {
    5: ("triangular_disk", "ring"),
    7: ("triangular_disk", "ring"),
    9: ("triangular_disk",),
    11: ("triangular_disk",),
    13: ("triangular_disk",),
    17: ("triangular_disk",),
    19: ("triangular_disk",),
    21: ("triangular_disk",),
    25: ("triangular_disk",),
    30: ("triangular_disk",),
}


def linear_chain(n_atoms: int, *, spacing: float = DEFAULT_SPACING) -> Tensor:
    """Return ``n_atoms`` positions along the x axis, centered at origin.

    Args:
        n_atoms: Number of atoms. Must be at least 1.
        spacing: Nearest-neighbor distance.

    Returns:
        Tensor of shape ``(n_atoms, 2)``.
    """
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")
    xs = (torch.arange(n_atoms, dtype=torch.float32) - (n_atoms - 1) / 2.0) * spacing
    ys = torch.zeros(n_atoms, dtype=torch.float32)
    return torch.stack([xs, ys], dim=-1)


def regular_polygon(n_atoms: int, *, spacing: float = DEFAULT_SPACING) -> Tensor:
    """Return ``n_atoms`` vertices of a regular polygon centered at origin.

    The polygon's circumradius is chosen so neighboring vertices sit at
    distance ``spacing``.

    Args:
        n_atoms: Number of vertices. Must be at least 2.
        spacing: Nearest-neighbor distance.

    Returns:
        Tensor of shape ``(n_atoms, 2)``.
    """
    if n_atoms < 2:
        raise ValueError(f"n_atoms must be >= 2, got {n_atoms}")
    # For a regular polygon: side = 2 * R * sin(pi/n_atoms)
    radius = spacing / (2.0 * math.sin(math.pi / n_atoms)) if n_atoms >= 3 else spacing / 2.0
    angles = torch.arange(n_atoms, dtype=torch.float32) * (2.0 * math.pi / n_atoms)
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)


def triangular_lattice_disk(
    n_atoms: int, *, spacing: float = DEFAULT_SPACING,
) -> Tensor:
    """Return the ``n_atoms`` atoms closest to origin on a triangular lattice.

    The function builds a candidate set of lattice points wide enough
    to contain ``n_atoms`` neighbors, sorts them by distance from the
    origin (with ties broken by polar angle for determinism), and
    returns the first ``n_atoms``.

    Args:
        n_atoms: Number of atoms.
        spacing: Lattice constant (nearest-neighbor distance).

    Returns:
        Tensor of shape ``(n_atoms, 2)``.
    """
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")

    # The triangular lattice in 2D: basis vectors a1 = (1, 0), a2 = (1/2, sqrt(3)/2).
    # The N atoms closest to origin lie inside a radius approximately
    # sqrt(N / pi) * spacing. We expand the search box generously.
    n_max = int(math.ceil(math.sqrt(n_atoms))) + 3
    a1 = torch.tensor([1.0, 0.0])
    a2 = torch.tensor([0.5, math.sqrt(3.0) / 2.0])
    points = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            p = (i * a1 + j * a2) * spacing
            points.append(p)
    pts = torch.stack(points, dim=0)
    distances = pts.norm(dim=-1)
    angles = torch.atan2(pts[:, 1], pts[:, 0])
    # Deterministic tie-breaking: sort by distance with a tiny angle
    # offset that breaks ties without affecting the primary ordering.
    sort_key = distances + 1.0e-6 * angles
    order = torch.argsort(sort_key, stable=True)
    return pts[order][:n_atoms].to(torch.float32)


def equilibrium_positions(
    n_atoms: int,
    motif: str = "triangular_disk",
    *,
    spacing: float = DEFAULT_SPACING,
) -> Tensor:
    """Dispatch to the named motif generator.

    Args:
        n_atoms: Number of atoms.
        motif: One of ``"linear"``, ``"ring"``, ``"triangular_disk"``.
        spacing: Nearest-neighbor distance.

    Returns:
        Tensor of shape ``(n_atoms, 2)``.
    """
    if motif == "linear":
        return linear_chain(n_atoms, spacing=spacing)
    if motif == "ring":
        return regular_polygon(n_atoms, spacing=spacing)
    if motif == "triangular_disk":
        return triangular_lattice_disk(n_atoms, spacing=spacing)
    raise ValueError(
        f"unknown motif {motif!r}. Valid motifs: linear, ring, triangular_disk."
    )


def valid_motifs(n_atoms: int) -> tuple[str, ...]:
    """Return the motifs available for the given ``n_atoms``.

    Falls back to ``("triangular_disk",)`` when ``n_atoms`` does not
    appear in ``VALID_MOTIFS_FOR_N``.
    """
    return VALID_MOTIFS_FOR_N.get(n_atoms, ("triangular_disk",))


__all__ = [
    "DEFAULT_SPACING",
    "VALID_MOTIFS_FOR_N",
    "equilibrium_positions",
    "linear_chain",
    "regular_polygon",
    "triangular_lattice_disk",
    "valid_motifs",
]
