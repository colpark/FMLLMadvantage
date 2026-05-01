# fmllm.physics

LJ Hamiltonian, MD integrator, cluster structures, and observables.
All quantities use reduced Lennard-Jones units (`epsilon = sigma = m = 1`).

## Files

- `lj_potential.py` - 12-6 Lennard-Jones pair potential and isotropic
  harmonic confinement. Computes total energy and per-atom forces with
  arbitrary leading batch dimensions. The minimum sits at
  `R_MIN = 2 ** (1/6)` with depth `-1`.
- `md.py` - vectorized velocity-Verlet integrator, equilibration
  helper with optional velocity-rescaling thermostat, and a
  Maxwell-Boltzmann velocity sampler that removes COM drift and
  optionally rescales to the equipartition target.
- `structures.py` - 2D cluster motif generators: `linear_chain`,
  `regular_polygon`, `triangular_lattice_disk`. The triangular-disk
  generator returns the `N` lattice points closest to the origin,
  which produces closed-shell clusters at `N = 7`, `13`, ... that
  serve as the 2D analogues of the 3D pentagonal-bipyramid and
  icosahedron motifs.
- `observables.py` - `pairwise_distances`, `pair_distance_histogram`
  (sum equals `N * (N - 1)` over the cluster diameter),
  `radial_distribution_function`, `kinetic_energies_per_atom`,
  `temperature_from_velocities`, and `rasterize_positions` (Gaussian
  blob render to a square grayscale image).
- `__init__.py` - re-exports the public API of all four modules.

## Conventions

- Positions and velocities have shape `(..., N, D)`. Functions
  broadcast over arbitrary leading batch dimensions.
- Forces equal accelerations under the unit-mass convention.
- The rasterizer image convention: `image[row, col]` maps to
  `(x = (col - W/2) * pixel_size, y = (H/2 - row) * pixel_size)`,
  so positive `y` lies in the upper half of the image.
