# Data Format

Phase 1 will fill this document with the on-disk schema for the
synthetic Lennard-Jones dataset and the trajectory format the
orchestrator produces.

The current draft fixes only a few invariants:

- Specimens carry a stable integer ID. The split assignments record IDs
  per split as YAML so a partial regeneration cannot scramble them.
- The dataset lives under `data/synthetic_lj_v1/` and is generated on
  the remote.
- The dataset directory holds an HDF5 store plus a `manifest.yaml`
  describing the generation parameters.
- Each specimen exposes three modalities:
  - 64 by 64 grayscale image with Gaussian-blob rasterization.
  - Length-200 radial distribution function.
  - 100-step MD trajectory snippet (positions and velocities per step).
- Specimens record N (atom count) from {5, 7, 9, 11, 13, 17, 19, 21,
  25, 30} and T (temperature) drawn log-uniform on [0.1, 2.0] LJ units.

The exact HDF5 group layout, dtype choices, and metadata fields
finalize in Phase 1 alongside the generator implementation.
