"""Velocity-Verlet integrator and Maxwell-Boltzmann velocity sampler.

This module provides a simple symplectic integrator for 2D molecular
dynamics. The integrator broadcasts over arbitrary leading batch
dimensions, so the caller can run multiple independent simulations in
parallel on a single GPU. The convention everywhere is unit mass per
atom in reduced LJ units.

Velocity-Verlet update (one step of size ``dt``)::

    v_half = v + 0.5 * dt * a
    r_new  = r + dt * v_half
    a_new  = forces_fn(r_new)
    v_new  = v_half + 0.5 * dt * a_new

The integrator caches the most recent acceleration so the inner loop
calls ``forces_fn`` exactly once per step.

Produces:
    Trajectories of positions and velocities for the requested number
    of recorded frames, plus a helper that initializes velocities at a
    target temperature with zero center-of-mass drift.

Depends on:
    torch.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

ForcesFn = Callable[[Tensor], tuple[Tensor, Tensor]]
"""Callable mapping ``positions`` to ``(potential_energy, forces)``."""


def maxwell_boltzmann_velocities(
    n_atoms: int,
    temperature: float,
    *,
    dim: int = 2,
    batch_shape: tuple[int, ...] = (),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    remove_com: bool = True,
    rescale_to_target: bool = True,
) -> Tensor:
    """Sample Maxwell-Boltzmann velocities at a given temperature.

    The function draws ``v ~ Normal(0, sqrt(temperature))`` per
    component, optionally subtracts the center-of-mass velocity, and
    optionally rescales the kinetic energy to the equipartition target
    ``0.5 * dim * (n_atoms - int(remove_com)) * temperature``.

    Args:
        n_atoms: Number of atoms.
        temperature: Target temperature in reduced LJ units.
        dim: Spatial dimension. The project uses 2.
        batch_shape: Optional leading batch dimensions, for example
            ``(B,)`` to draw ``B`` independent velocity sets.
        device: Target device.
        dtype: Floating-point dtype.
        generator: Optional torch generator for reproducibility.
        remove_com: When True, subtract the per-batch center-of-mass
            velocity so the cluster has zero net momentum.
        rescale_to_target: When True, rescale velocities so kinetic
            energy hits the equipartition target exactly. Use this for
            deterministic temperature initialization.

    Returns:
        A tensor of shape ``batch_shape + (n_atoms, dim)``.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")

    shape = (*batch_shape, n_atoms, dim)
    velocities = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    velocities = velocities * (temperature ** 0.5)

    if remove_com and n_atoms > 1:
        com = velocities.mean(dim=-2, keepdim=True)
        velocities = velocities - com

    if rescale_to_target and temperature > 0 and n_atoms > 1:
        free_dof = dim * (n_atoms - (1 if remove_com else 0))
        ke_target = 0.5 * free_dof * temperature
        ke_actual = 0.5 * (velocities * velocities).sum(dim=(-1, -2), keepdim=True)
        ke_actual = ke_actual.squeeze(-1)
        # Avoid division by zero when n_atoms is 1 or temperature is 0.
        scale = torch.where(
            ke_actual > 0,
            (ke_target / ke_actual.clamp_min(1e-30)) ** 0.5,
            torch.ones_like(ke_actual),
        )
        velocities = velocities * scale.unsqueeze(-1)

    return velocities


def velocity_verlet_step(
    positions: Tensor,
    velocities: Tensor,
    accelerations: Tensor,
    forces_fn: ForcesFn,
    dt: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Advance one velocity-Verlet step.

    The caller passes the current acceleration so the function evaluates
    ``forces_fn`` exactly once per step.

    Args:
        positions: Current positions, shape ``(..., N, D)``.
        velocities: Current velocities, same shape.
        accelerations: Current accelerations (same as forces under unit
            mass), same shape.
        forces_fn: Callable mapping ``positions`` to ``(energy, forces)``.
        dt: Timestep.

    Returns:
        ``(new_positions, new_velocities, new_accelerations, new_potential_energy)``.
    """
    v_half = velocities + 0.5 * dt * accelerations
    new_positions = positions + dt * v_half
    new_potential, new_accelerations = forces_fn(new_positions)
    new_velocities = v_half + 0.5 * dt * new_accelerations
    return new_positions, new_velocities, new_accelerations, new_potential


def run_md(
    positions: Tensor,
    velocities: Tensor,
    forces_fn: ForcesFn,
    *,
    dt: float,
    n_steps: int,
    record_every: int = 1,
    record_initial: bool = True,
) -> dict[str, Tensor]:
    """Run a molecular-dynamics simulation and record the trajectory.

    Args:
        positions: Initial positions, shape ``(..., N, D)``.
        velocities: Initial velocities, same shape.
        forces_fn: Callable mapping ``positions`` to ``(energy, forces)``.
        dt: Timestep.
        n_steps: Number of velocity-Verlet steps to integrate.
        record_every: Stride at which to record frames. ``1`` records
            every step.
        record_initial: When True, the trajectory's first frame is the
            input state at ``t = 0``.

    Returns:
        A dict with keys ``positions``, ``velocities``, ``potential_energies``,
        ``kinetic_energies``, each with leading dimension equal to the
        number of recorded frames.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    if record_every < 1:
        raise ValueError(f"record_every must be >= 1, got {record_every}")

    initial_pe, accelerations = forces_fn(positions)

    pos_frames: list[Tensor] = []
    vel_frames: list[Tensor] = []
    pe_frames: list[Tensor] = []

    if record_initial:
        pos_frames.append(positions.clone())
        vel_frames.append(velocities.clone())
        pe_frames.append(initial_pe.clone())

    current_pe = initial_pe
    for step in range(1, n_steps + 1):
        positions, velocities, accelerations, current_pe = velocity_verlet_step(
            positions, velocities, accelerations, forces_fn, dt,
        )
        if step % record_every == 0:
            pos_frames.append(positions.clone())
            vel_frames.append(velocities.clone())
            pe_frames.append(current_pe.clone())

    pos_traj = torch.stack(pos_frames, dim=0)
    vel_traj = torch.stack(vel_frames, dim=0)
    pe_traj = torch.stack(pe_frames, dim=0)
    ke_traj = 0.5 * (vel_traj * vel_traj).sum(dim=(-1, -2))

    return {
        "positions": pos_traj,
        "velocities": vel_traj,
        "potential_energies": pe_traj,
        "kinetic_energies": ke_traj,
    }


def equilibrate(
    positions: Tensor,
    velocities: Tensor,
    forces_fn: ForcesFn,
    *,
    dt: float,
    n_steps: int,
    rescale_temperature: float | None = None,
    rescale_every: int = 0,
) -> tuple[Tensor, Tensor]:
    """Run MD for ``n_steps`` and return only the final state.

    This helper avoids allocating the trajectory arrays. Optionally
    rescales velocities to a target temperature every ``rescale_every``
    steps, which is a crude thermostat that keeps the equilibration
    near the requested temperature without recording the trajectory.

    Args:
        positions: Initial positions.
        velocities: Initial velocities.
        forces_fn: Callable mapping ``positions`` to ``(energy, forces)``.
        dt: Timestep.
        n_steps: Number of velocity-Verlet steps.
        rescale_temperature: Optional temperature target for the simple
            velocity-rescaling thermostat. ``None`` disables rescaling.
        rescale_every: Stride between rescaling events. ``0`` disables.

    Returns:
        ``(final_positions, final_velocities)``.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")

    if n_steps == 0:
        return positions, velocities

    _, accelerations = forces_fn(positions)
    n_atoms = positions.shape[-2]
    dim = positions.shape[-1]
    free_dof = dim * (n_atoms - 1)  # zero-COM constraint

    for step in range(1, n_steps + 1):
        positions, velocities, accelerations, _ = velocity_verlet_step(
            positions, velocities, accelerations, forces_fn, dt,
        )
        if (
            rescale_temperature is not None
            and rescale_every > 0
            and step % rescale_every == 0
            and free_dof > 0
        ):
            ke = 0.5 * (velocities * velocities).sum(dim=(-1, -2), keepdim=True)
            ke_target = 0.5 * free_dof * rescale_temperature
            scale = torch.where(
                ke > 0,
                (ke_target / ke.clamp_min(1e-30)) ** 0.5,
                torch.ones_like(ke),
            )
            velocities = velocities * scale

    return positions, velocities


__all__ = [
    "ForcesFn",
    "equilibrate",
    "maxwell_boltzmann_velocities",
    "run_md",
    "velocity_verlet_step",
]
