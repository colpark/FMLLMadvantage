# Data Format

This document captures the on-disk layout of the synthetic Lennard-
Jones dataset that `fmllm.data.generator` produces.

All quantities use reduced LJ units (`epsilon = sigma = m = 1`).

## Directory layout

```
data/synthetic_lj_v1/
├── specimens.h5     # the full dataset
├── splits.yaml      # train / holdout assignments
└── manifest.yaml    # generation parameters and provenance
```

## HDF5 file

`specimens.h5` is a single HDF5 file with one row per specimen. The
generator pads atom-axis dimensions to `max(n_choices) = 30` so all
arrays have fixed shape, and stores the per-specimen atom count plus a
boolean mask. Padded entries hold zeros.

### Datasets

| Name | Shape | dtype | Description |
|------|-------|-------|-------------|
| `images` | `(num_specimens, image_size, image_size)` | float32 | Rasterized 64x64 grayscale images of the final equilibrated frame |
| `rdfs` | `(num_specimens, rdf_bins)` | float32 | Radial distribution function `g(r)` over `[0, rdf_r_max]` |
| `traj_positions` | `(num_specimens, n_steps + 1, max_n_atoms, 2)` | float32 | Position trajectory snippet, padded along the atom axis |
| `traj_velocities` | `(num_specimens, n_steps + 1, max_n_atoms, 2)` | float32 | Velocity trajectory snippet, padded along the atom axis |
| `equilibrium_positions` | `(num_specimens, max_n_atoms, 2)` | float32 | Initial post-equilibration positions, before recording the snippet |
| `atom_counts` | `(num_specimens,)` | int32 | Number of real atoms `N` for each specimen |
| `temperatures` | `(num_specimens,)` | float32 | Specimen temperature `T` |
| `motif_ids` | `(num_specimens,)` | int16 | Index into `motif_names` (root attribute) |
| `seeds` | `(num_specimens,)` | int64 | Master per-specimen seed used during generation |

`n_steps` equals `md_steps_per_specimen` from the config, so the
trajectory length is `n_steps + 1` because the snippet records the
initial state.

### Root attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `max_n_atoms` | int | Padding width along the atom axis |
| `image_size` | int | Width and height of the rasterized image |
| `image_pixel_size_lj` | float | Pixel width in LJ units |
| `image_blur_radius_lj` | float | Gaussian-blob standard deviation in LJ units |
| `image_noise_std` | float | Additive Gaussian noise standard deviation |
| `rdf_bins` | int | Number of RDF bins |
| `rdf_r_max` | float | Upper edge of the RDF domain |
| `md_dt` | float | MD timestep |
| `md_steps_per_specimen` | int | Length of the recorded trajectory snippet |
| `md_equilibration_steps` | int | Number of equilibration steps before recording |
| `confinement_k` | float | Harmonic-confinement spring constant |
| `motif_names` | bytes array | Stable mapping `motif_id -> name` |

## splits.yaml

Held-out partitioning lives in a YAML file alongside the HDF5. The
schema is:

```yaml
train: [<sorted list of specimen IDs>]
holdout:
  in_n_in_t: [<sorted list of specimen IDs>]
  in_n_ood_t: [<sorted list of specimen IDs>]
  ood_n_in_t: [<sorted list of specimen IDs>]
  ood_n_ood_t: [<sorted list of specimen IDs>]
meta:
  num_specimens: 50000
  num_holdout: 10000
  n_in_distribution: [5, 7, 9, 11, 13]
  t_in_distribution_max: 1.0
  seed: 1234
```

The four holdout cells partition the held-out specimens by atom-count
distribution (`in_n` vs `ood_n`) and by temperature distribution
(`in_t` vs `ood_t`).

## manifest.yaml

The manifest follows the project-wide schema in
`fmllm.utils.manifests`. The `inputs` field records the config path;
`config` records the resolved `DatasetConfig` plus runtime metadata
(run ID, device, batch size, total specimens generated); `extra`
records the per-`N` distribution and elapsed wall-clock time.

## Specimen IDs

Specimen IDs are non-negative integers starting at zero. The ID equals
the row index in the HDF5 datasets, which lets the splits YAML (and
any downstream code) refer to specimens by integer without consulting
the file. Re-running the generator with the same `master_seed` and
`num_specimens` reproduces the same `(N, T, motif)` per ID.
