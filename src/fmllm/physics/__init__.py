"""LJ Hamiltonian, MD integrator, structures, and observables.

The subpackage exposes the building blocks the dataset generator and
the foundation models share. All distances and energies use reduced
Lennard-Jones units (``epsilon = sigma = m = 1``).

Public modules:
    lj_potential: pair potential plus harmonic confinement.
    md: velocity-Verlet integrator and Maxwell-Boltzmann sampler.
    structures: 2D cluster motif generators.
    observables: pair distances, RDF, kinetic energies, rasterizer.
"""

from fmllm.physics.lj_potential import (
    R_MIN,
    harmonic_confinement_energy_and_forces,
    lj_pair_energy_and_forces,
    potential_at_distance,
    total_energy_and_forces,
)
from fmllm.physics.md import (
    ForcesFn,
    equilibrate,
    maxwell_boltzmann_velocities,
    run_md,
    velocity_verlet_step,
)
from fmllm.physics.observables import (
    kinetic_energies_per_atom,
    pair_distance_histogram,
    pairwise_distances,
    radial_distribution_function,
    rasterize_positions,
    temperature_from_velocities,
)
from fmllm.physics.structures import (
    DEFAULT_SPACING,
    VALID_MOTIFS_FOR_N,
    equilibrium_positions,
    linear_chain,
    regular_polygon,
    triangular_lattice_disk,
    valid_motifs,
)

__all__ = [
    "DEFAULT_SPACING",
    "ForcesFn",
    "R_MIN",
    "VALID_MOTIFS_FOR_N",
    "equilibrate",
    "equilibrium_positions",
    "harmonic_confinement_energy_and_forces",
    "kinetic_energies_per_atom",
    "linear_chain",
    "lj_pair_energy_and_forces",
    "maxwell_boltzmann_velocities",
    "pair_distance_histogram",
    "pairwise_distances",
    "potential_at_distance",
    "radial_distribution_function",
    "rasterize_positions",
    "regular_polygon",
    "run_md",
    "temperature_from_velocities",
    "total_energy_and_forces",
    "triangular_lattice_disk",
    "valid_motifs",
    "velocity_verlet_step",
]
