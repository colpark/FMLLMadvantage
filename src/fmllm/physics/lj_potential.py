"""Lennard-Jones pair potential and harmonic confinement.

This module implements the 12-6 Lennard-Jones pair potential and an
isotropic harmonic confinement, both in PyTorch so they run on GPU and
broadcast over arbitrary leading batch dimensions. Reduced units apply
throughout: ``epsilon = sigma = m = 1``. Time then carries units of
``sqrt(m * sigma**2 / epsilon) = 1``, and temperature carries units of
``epsilon / k_B = 1``.

Key formulas (reduced units):

    U_LJ(r)     = 4 * (r**-12 - r**-6)
    F_LJ(r)     = 24 * r**-7 * (2 * r**-6 - 1)            (magnitude)
    U_conf(r)   = 0.5 * k * |r - r0|**2
    F_conf(r)   = -k * (r - r0)

The minimum of ``U_LJ`` sits at ``r_min = 2**(1/6)`` with depth ``-1``.
``lj_pair_energy_and_forces`` ignores the diagonal (self-interactions)
and counts each pair once when summing the energy.

Produces:
    Total potential energy as a scalar per leading batch index, and
    per-atom forces with the same shape as the input positions.

Depends on:
    torch.
"""

from __future__ import annotations

import torch
from torch import Tensor

R_MIN = 2.0 ** (1.0 / 6.0)
"""Equilibrium pair distance for the 12-6 LJ potential in reduced units."""


def lj_pair_energy_and_forces(
    positions: Tensor,
    *,
    r_clip: float = 1.0e-3,
) -> tuple[Tensor, Tensor]:
    """Compute the LJ pair potential energy and per-atom forces.

    Args:
        positions: Tensor of shape ``(..., N, D)``. The leading dimensions
            broadcast over independent simulations. ``D`` must be 2 or 3.
        r_clip: Lower clamp on pair distance squared. Prevents division
            by zero when two atoms collide. The clamp value is small
            enough that any physical configuration stays unaffected.

    Returns:
        A tuple ``(energy, forces)`` where ``energy`` has shape
        ``(...,)`` and ``forces`` has the same shape as ``positions``.
    """
    n_atoms = positions.shape[-2]
    diff = positions.unsqueeze(-2) - positions.unsqueeze(-3)  # (..., N, N, D)
    r2 = (diff * diff).sum(dim=-1)  # (..., N, N)

    eye_mask = torch.eye(n_atoms, dtype=torch.bool, device=positions.device)
    # Replace the diagonal with 1.0 so the inv_r2 calculation below stays
    # finite. The eye_mask zeros the result on the diagonal afterwards.
    r2_safe = torch.where(eye_mask, torch.ones_like(r2), r2.clamp_min(r_clip))

    inv_r2 = 1.0 / r2_safe
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6

    pair_energy = 4.0 * (inv_r12 - inv_r6)
    pair_energy = torch.where(eye_mask, torch.zeros_like(pair_energy), pair_energy)
    energy = 0.5 * pair_energy.sum(dim=(-1, -2))

    # F_i = sum_{j != i} pair_factor_ij * (r_i - r_j)
    # pair_factor = -(dU/dr) / r = 24 * r**-8 * (2 * r**-6 - 1)
    #             = 24 * inv_r2**4 * (2 * inv_r6 - 1)
    pair_factor = 24.0 * (inv_r2 * inv_r2 * inv_r2 * inv_r2) * (2.0 * inv_r6 - 1.0)
    pair_factor = torch.where(eye_mask, torch.zeros_like(pair_factor), pair_factor)

    forces = (pair_factor.unsqueeze(-1) * diff).sum(dim=-2)
    return energy, forces


def harmonic_confinement_energy_and_forces(
    positions: Tensor,
    *,
    k: float = 0.0,
    center: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the energy and forces from an isotropic harmonic well.

    The potential is ``0.5 * k * |r - center|**2`` per atom. ``k = 0``
    disables the confinement, which is convenient for the energy-
    conservation test on a pure-LJ system.

    Args:
        positions: Tensor of shape ``(..., N, D)``.
        k: Spring constant. The default ``0.0`` disables the well.
        center: Optional center of the well with shape broadcastable to
            ``(..., 1, D)``. ``None`` defaults to the origin.

    Returns:
        A tuple ``(energy, forces)`` matching the input shape.
    """
    if k == 0.0:
        energy = torch.zeros(positions.shape[:-2], device=positions.device, dtype=positions.dtype)
        forces = torch.zeros_like(positions)
        return energy, forces

    if center is None:
        offset = positions
    else:
        offset = positions - center

    energy = 0.5 * k * (offset * offset).sum(dim=(-1, -2))
    forces = -k * offset
    return energy, forces


def total_energy_and_forces(
    positions: Tensor,
    *,
    k_conf: float = 0.0,
    center: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Combined LJ pair plus harmonic confinement energy and forces.

    Args:
        positions: Tensor of shape ``(..., N, D)``.
        k_conf: Spring constant for the harmonic well. ``0.0`` disables it.
        center: Optional well center.

    Returns:
        A tuple ``(energy, forces)`` matching the input shape.
    """
    e_lj, f_lj = lj_pair_energy_and_forces(positions)
    if k_conf == 0.0:
        return e_lj, f_lj
    e_conf, f_conf = harmonic_confinement_energy_and_forces(
        positions, k=k_conf, center=center,
    )
    return e_lj + e_conf, f_lj + f_conf


def potential_at_distance(r: float | Tensor) -> Tensor:
    """Convenience helper: scalar U_LJ(r) for testing and plotting."""
    r_t = torch.as_tensor(r, dtype=torch.float64)
    inv_r6 = r_t.pow(-6)
    inv_r12 = inv_r6 * inv_r6
    return 4.0 * (inv_r12 - inv_r6)


__all__ = [
    "R_MIN",
    "lj_pair_energy_and_forces",
    "harmonic_confinement_energy_and_forces",
    "total_energy_and_forces",
    "potential_at_distance",
]
