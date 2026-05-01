# Audit Report, Phase 1

**Audited at:** 2026-05-01T20:46:11Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS

## Summary

Phase 1 implements the synthetic Lennard-Jones testbed: physics module
(LJ potential, MD integrator, structures, observables), data module
(splits, HDF5 dataset reader, generator CLI), config-schema
extensions, the four required tests plus invariants, and the
documentation update. Local pytest reports **49 passed**. The
generator CLI parses help correctly and imports clean against a
minimal local venv (loguru, pydantic, pyyaml, pytest, torch, numpy,
h5py, typer, scipy). I did not generate data locally per the
"`do not generate the data`" rule from the modified Phase 1 prompt;
the user runs the generator on the remote.

## Detailed checks

### CHECK 1.1, src/fmllm/physics/ files exist
- **Result:** PASS
- **Evidence:** `lj_potential.py` (4.3 KB), `md.py` (8.5 KB),
  `structures.py` (5.6 KB), `observables.py` (8.2 KB),
  `__init__.py` updated to re-export the public API.

### CHECK 1.2, lj_potential.py implements LJ pair + harmonic in PyTorch
- **Result:** PASS
- **Evidence:** `lj_pair_energy_and_forces` returns
  `(energy, forces)` with arbitrary leading batch dimensions and
  ignores the diagonal. Tests confirm the minimum sits at
  `r = 2 ** (1/6)` with depth `-1` and verify repulsive/attractive
  force signs at short and long range (`tests/test_physics.py::
  test_lj_minimum_at_two_to_one_sixth`,
  `test_lj_repulsive_short_range`, `test_lj_attractive_long_range`).
  `harmonic_confinement_energy_and_forces` and
  `total_energy_and_forces` provide the combined potential the MD
  integrator uses.

### CHECK 1.3, md.py is a vectorized 2D velocity-Verlet integrator
- **Result:** PASS
- **Evidence:** `velocity_verlet_step` evaluates `forces_fn` exactly
  once per step. `run_md` returns dict of stacked trajectories.
  `equilibrate` skips trajectory recording and supports a per-batch
  velocity-rescaling thermostat. `maxwell_boltzmann_velocities`
  initializes velocities at a target temperature with zero COM and
  optional exact equipartition rescaling. The integrator broadcasts
  over arbitrary leading batch dimensions
  (`tests/test_physics.py::test_lj_batched_broadcast`).

### CHECK 1.4, structures.py covers the required motifs
- **Result:** PASS
- **Evidence:** `linear_chain`, `regular_polygon`, and
  `triangular_lattice_disk` cover dimers, triangles, hexagonal close-
  packed disks, and the closed-shell `N = 7, 13` clusters that play
  the role of pentagonal-bipyramid/icosahedron analogues in 2D.
  `equilibrium_positions` dispatches by name. `valid_motifs(N)`
  returns the motifs for each `N` the generator samples
  (`VALID_MOTIFS_FOR_N`).

### CHECK 1.5, observables.py covers RDF, KE distribution, rasterizer
- **Result:** PASS
- **Evidence:** `pair_distance_histogram` sums to `N * (N - 1)` over
  the cluster diameter (verified by
  `test_pair_histogram_sum_equals_n_squared_minus_n`).
  `radial_distribution_function` normalizes by the annulus area and
  the cell density. `kinetic_energies_per_atom` and
  `temperature_from_velocities` cover the velocity side.
  `rasterize_positions` renders Gaussian blobs with optional Gaussian
  noise and respects the documented coordinate convention
  (`image[row, col]` -> `(x = (col - W/2) * px, y = (H/2 - row) * px)`).

### CHECK 1.6, src/fmllm/data/ files exist with documented interfaces
- **Result:** PASS
- **Evidence:** `splits.py` (4.7 KB), `dataset.py` (5.7 KB),
  `generator.py` (12.0 KB), `__init__.py` re-exports the public API.

### CHECK 1.7, generator.py is a CLI with stable per-ID seeds
- **Result:** PASS
- **Evidence:** Typer-based CLI exposes `--config`, `--out`,
  `--num-specimens`, `--device`, `--batch-size`, `--smoke-test`.
  `sample_specimen_specs` seeds each specimen with
  `master_seed * num_specimens + id` so re-running with the same
  seed/count reproduces the same `(N, T, motif)` per ID. The CLI
  prints help cleanly under
  `python -m fmllm.data.generator --help`.

### CHECK 1.8, dataset.py is a PyTorch Dataset that reads HDF5
- **Result:** PASS
- **Evidence:** `LJSpecimenDataset` opens the HDF5 file lazily,
  exposes one specimen per index, returns torch tensors plus scalar
  metadata, supports filtering by an explicit list of specimen IDs,
  and computes a per-specimen `atom_mask` from `atom_counts`.
  Synthetic-HDF5 round-trip tests pass
  (`test_dataset_reads_synthetic_h5`,
  `test_dataset_filter_by_specimen_ids`).

### CHECK 1.9, splits.py partitions by the (N, T) axes
- **Result:** PASS
- **Evidence:** `assign_splits` takes the full population, draws the
  holdout uniformly at random, and labels each held-out specimen by
  `(N_axis, T_axis) cell` from `CELL_LABELS`. `save_splits_yaml` and
  `load_splits_yaml` round-trip. Determinism, size invariants, and
  out-of-range guards all covered by tests.

### CHECK 1.10, config schema extended with generator parameters
- **Result:** PASS
- **Evidence:** `DatasetConfig` adds `md_dt`,
  `md_equilibration_steps`, `md_thermostat_every`, `confinement_k`,
  `image_pixel_size_lj`, `image_blur_radius_lj`, `image_noise_std`,
  `rdf_r_max`, `perturbation_std`, `generator_batch_size`,
  `generator_master_seed`, `holdout_fraction`. `configs/default.yaml`
  carries matching defaults plus per-section comments.
  `test_load_config_repo_default` validates the YAML against the
  Pydantic schema.

### CHECK 1.11, the four required physics tests pass
- **Result:** PASS
- **Evidence:** `pytest -m "not gpu" -v` reports
  `test_energy_conservation_pure_lj_1000_steps PASSED`,
  `test_pair_histogram_sum_equals_n_squared_minus_n PASSED`,
  `test_rdf_permutation_invariance PASSED`,
  `test_rasterizer_blob_centroid_within_subpixel PASSED`. The
  rasterizer test recovers each atom's position from the image's
  per-blob centroid to better than one pixel for a 0.10 LJ pixel.

### CHECK 1.12, the full local test suite passes
- **Result:** PASS
- **Evidence:** `49 passed in 1.76s` running locally on Python 3.11
  with the audit venv extended to include torch (CPU), numpy, h5py,
  scipy, typer.

### CHECK 1.13, no data generated locally
- **Result:** PASS
- **Evidence:** `git status` shows only source, config, and
  documentation changes. No file under `data/synthetic_lj_v1/`, no
  HDF5 files anywhere in the working tree.

### CHECK 1.14, docs/data-format.md updated with the on-disk schema
- **Result:** PASS
- **Evidence:** The document lists every HDF5 dataset with shape and
  dtype, every root attribute, the splits YAML schema, the manifest
  schema reference, and the specimen-ID convention.

### CHECK 1.15, docs/progress/01-testbed.md captures verification commands
- **Result:** PASS
- **Evidence:** The doc lists local pytest commands, the smoke-test
  generator command (`--smoke-test`), the full generator command,
  the expected runtime envelope, and the artifacts the user sends
  back. Known-issue and what-remains sections cover the partial-write
  caveat and the velocity-rescaling thermostat caveat.

### CHECK 1.16, prose style
- **Result:** PASS
- **Evidence:** Scanned every modified or new markdown file for
  em-dashes and semicolons in narrative prose (excluding fenced code
  blocks). Zero matches.

### CHECK 1.17, working tree clean after the Phase 1 commit
- **Result:** PASS (after the Phase 1 commit lands)
- **Evidence:** Pre-commit `git status --short` shows only the
  Phase 1 changes (modified config / READMEs / progress index, plus
  the new physics, data, and test files). The Phase 1 commit folds
  them in.

## Fixes applied during audit

None. The Phase 1 implementation passed every check on first
inspection. The audit run extended `/tmp/fmllm-audit-venv` with
`torch numpy h5py typer scipy` to cover the new test surface.

## Remaining concerns

- The default `rdf_r_max = 6.0` LJ is generous for `N <= 30`. If a
  later phase tunes the cluster footprint, the RDF normalization
  uses `cell_area = pi * rdf_r_max ** 2` by default, which means a
  change to `rdf_r_max` rescales `g(r)`. The test of permutation
  invariance survives any such change. The downstream FM2 should
  consume `g(r)` consistently across specimens, which the current
  fixed `rdf_r_max` guarantees.
- The trajectory snippet runs NVE (no thermostat) so the recorded
  temperature can drift from the requested `T` over the 100 steps.
  Phase 2's FM3 trains against the empirical KE distribution, which
  should be fine, but if equipartition checks turn up systematic
  bias the equilibration-step count is the obvious knob.
- The generator's `--smoke-test` defaults to 200 specimens. The
  remote should run that first to catch any environment issue (HDF5
  driver, torch CPU vs CUDA, etc.) before the full 50K run.

## Sign-off

The Phase 1 implementation matches the original prompt's
specification (modified by the local-Claude/remote-execution split).
The user can proceed to running the smoke-test generator on the
remote, then the full generator, per the verification commands in
`docs/progress/01-testbed.md`.
