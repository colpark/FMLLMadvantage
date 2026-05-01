# Phase 1: Synthetic Lennard-Jones Testbed

## What I built

### Physics (`src/fmllm/physics/`)

- `lj_potential.py` - 12-6 Lennard-Jones pair potential and isotropic
  harmonic confinement, both in PyTorch with arbitrary leading batch
  dimensions. `total_energy_and_forces(positions, k_conf=...)` is the
  canonical interface MD uses.
- `md.py` - vectorized velocity-Verlet integrator with one
  `forces_fn` evaluation per step. `equilibrate` runs MD without
  recording a trajectory and supports a simple per-batch
  velocity-rescaling thermostat. `maxwell_boltzmann_velocities`
  initializes velocities at a target temperature with zero
  center-of-mass drift and optional exact equipartition rescaling.
- `structures.py` - `linear_chain`, `regular_polygon`,
  `triangular_lattice_disk`, plus `equilibrium_positions` and
  `valid_motifs`. The triangular-disk generator returns the `N` atoms
  closest to the origin on a 2D triangular lattice with
  `r_min = 2 ** (1/6)` spacing. `N = 7` and `N = 13` produce
  closed-shell clusters that play the role of the 2D analogues of the
  3D pentagonal-bipyramid and icosahedron motifs.
- `observables.py` - `pairwise_distances`, `pair_distance_histogram`
  (sum equals `N * (N - 1)` over the cluster diameter),
  `radial_distribution_function`, `kinetic_energies_per_atom`,
  `temperature_from_velocities`, `rasterize_positions` (Gaussian-blob
  rasterizer with optional Gaussian noise).

### Data (`src/fmllm/data/`)

- `splits.py` - `assign_splits` partitions specimens into train and
  holdout subsets. Held-out specimens carry a per-cell label across
  the four `(N_axis, T_axis)` cells. `save_splits_yaml` and
  `load_splits_yaml` round-trip the assignment to disk.
- `dataset.py` - `LJSpecimenDataset` is an HDF5-backed PyTorch
  Dataset. The reader supports filtering by an explicit list of
  specimen IDs.
- `generator.py` - the CLI script. Samples `(N, T, motif)` per
  specimen with deterministic per-ID seeds, groups specimens by `N`,
  runs equilibration plus a recorded trajectory snippet on each
  group, then writes one HDF5 file plus `splits.yaml` and
  `manifest.yaml` to the output directory. `--smoke-test` runs a
  200-specimen subset for an end-to-end check before committing to
  the full 50K run.

### Configuration

- Extended `DatasetConfig` in `src/fmllm/utils/config.py` with the
  generator parameters: `md_dt`, `md_equilibration_steps`,
  `md_thermostat_every`, `confinement_k`, `image_pixel_size_lj`,
  `image_blur_radius_lj`, `image_noise_std`, `rdf_r_max`,
  `perturbation_std`, `generator_batch_size`, `generator_master_seed`,
  `holdout_fraction`.
- Updated `configs/default.yaml` with matching values plus per-section
  comments explaining the generator behavior.

### Tests

- `tests/test_physics.py` - the four required tests plus invariants:
  - `test_energy_conservation_pure_lj_1000_steps` - max energy drift
    below `1e-3` over 1000 velocity-Verlet steps with `dt = 0.002`.
  - `test_pair_histogram_sum_equals_n_squared_minus_n` - pair distance
    histogram sums to `N * (N - 1)` for `N = 13`.
  - `test_rdf_permutation_invariance` - `g(r)` is identical under a
    random permutation of atoms.
  - `test_rasterizer_blob_centroid_within_subpixel` - per-blob
    centroid recovers atom position to better than one pixel.
  - Plus: LJ minimum at `r = 2 ** (1/6)`, repulsive/attractive force
    signs, batched broadcasting, MB target temperature and zero-COM,
    structure-generator counts and nearest-neighbor spacing.
- `tests/test_data.py` - splits cell labeling, deterministic
  assignment, size invariants, YAML round trip, dataset reader on a
  tiny synthetic HDF5 file.

### Documentation

- `docs/data-format.md` - HDF5 layout, dataset attributes, splits YAML
  schema, manifest schema, specimen-ID convention.
- `src/fmllm/physics/README.md` and `src/fmllm/data/README.md` updated
  with file lists and conventions.

## What the user runs to verify Phase 1

### Local laptop (no GPU)

```
cd FMLLMadvantage
uv sync --extra dev   # if not already done
uv run pytest -m "not gpu" -v
```

Expect every test in `tests/test_utils.py`, `tests/test_physics.py`,
and `tests/test_data.py` to pass. The full local run takes seconds.

### Remote 4xH100 host

#### Step 1. Smoke-test the generator

A 200-specimen subset confirms the generator runs end to end without
touching the full 50K budget:

```
CUDA_VISIBLE_DEVICES=0 uv run python -m fmllm.data.generator \
    --config configs/default.yaml \
    --out runs/data-smoke \
    --smoke-test
```

Expect:
- `runs/data-smoke/specimens.h5` of order tens of MB.
- `runs/data-smoke/splits.yaml` summing to 200 specimens.
- `runs/data-smoke/manifest.yaml` describing the run.
- Wall-clock runtime around 1-2 minutes on H100.

Then run the suite again to confirm nothing regressed:

```
uv run pytest -m "not gpu" -v
```

#### Step 2. Generate the full dataset

```
CUDA_VISIBLE_DEVICES=0 uv run python -m fmllm.data.generator \
    --config configs/default.yaml \
    --out data/synthetic_lj_v1
```

Expect:
- Wall-clock runtime around 20 to 40 minutes on a single H100. The
  generator processes one batch at a time, grouped by `N`, with a
  default `generator_batch_size` of 256.
- `data/synthetic_lj_v1/specimens.h5` of order 3 to 4 GB.
- `data/synthetic_lj_v1/splits.yaml` summing to 50,000 specimens with
  10,000 in the holdout (split across the four `(N_axis, T_axis)` cells).
- `data/synthetic_lj_v1/manifest.yaml` recording the resolved config,
  per-`N` distribution, and elapsed time.

## What to send back

- Full stdout from the smoke-test generator run.
- Full stdout from the full generator run, or at least the final
  `Done in ...` line and the per-batch progress excerpts.
- `runs/<run_id>/run.log` for the smoke-test (find the run ID in the
  generator's first log line).
- `data/synthetic_lj_v1/manifest.yaml` and
  `data/synthetic_lj_v1/splits.yaml`. Manifest YAMLs commit back to
  the repo via the `data/**/*.yaml` gitignore exception.
- The `atom_counts` and `temperatures` summary the generator logs.
  Any unexpected skew in the distribution (more or fewer atoms in any
  N bucket than 1/10 of the total) flags a sampling bug.

## Known issues to flag

- The generator allocates the HDF5 store with all rows pre-allocated
  to the right shape. Crashes mid-run leave a partially populated
  file. Re-running overwrites it cleanly.
- The simple velocity-rescaling thermostat is a crude tool. It keeps
  the equilibration near the requested temperature on average but
  introduces small artifacts in the velocity distribution. The
  trajectory snippet itself runs without thermostatting (NVE), so the
  recorded velocities follow the post-equilibration microcanonical
  ensemble.
- The default RDF normalization uses `cell_area = pi * rdf_r_max**2`,
  which is convention-dependent. The number that matters for FM2 is
  consistent across specimens, which this convention satisfies. The
  Phase 1 tests cover the histogram normalization directly.

## What remains for Phase 2

- Implement the three foundation models under `src/fmllm/fms/`:
  - `fm1_image/` Vision Transformer (10-30M params) for the 64x64
    grayscale image, predicting atom count and atom positions with a
    DETR-style set-prediction head.
  - `fm2_rdf/` 1D Transformer (5-15M params) for the length-200
    `g(r)`, predicting coarse-grained energy per atom.
  - `fm3_traj/` Trajectory Transformer (10-25M params) for the
    100-step MD snippet, predicting Gamma moments of the kinetic
    energy distribution.
- Add `scripts/train_fm.py` as a unified CLI that dispatches across
  FMs and supports `--calibrate-only` for the post-training conformal
  step.
- Add tests in `tests/test_fms.py` that exercise forward-pass shapes
  and physics-constraint losses on synthetic inputs.
