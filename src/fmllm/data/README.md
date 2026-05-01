# fmllm.data

Synthetic Lennard-Jones dataset generation and loading.

## Files

- `generator.py` - CLI script that produces the full dataset. Samples
  ``(N, T, motif)`` per specimen with deterministic per-ID seeds,
  groups specimens by ``N``, runs MD equilibration plus a recorded
  trajectory snippet, then writes one HDF5 file plus a manifest YAML
  and a splits YAML to the output directory.
- `dataset.py` - `LJSpecimenDataset`, an HDF5-backed PyTorch Dataset.
  Supports filtering by an explicit list of specimen IDs, which is how
  the splits machinery selects a train or holdout view of the same
  underlying file.
- `splits.py` - `assign_splits`, `save_splits_yaml`, `load_splits_yaml`.
  Held-out specimens are labeled by ``(N_axis, T_axis)`` cell so the
  evaluation framework can report per-cell metrics.

## On-disk format

See `docs/data-format.md` for the HDF5 layout and the splits YAML
schema.

## CLI

```
uv run python -m fmllm.data.generator \
    --config configs/default.yaml \
    --out data/synthetic_lj_v1
```

Useful flags:
- `--smoke-test` runs a 200-specimen subset for end-to-end validation.
- `--num-specimens N` overrides the count from the config.
- `--device cuda` or `--device cpu` selects the compute device.
- `--batch-size B` overrides the per-`N` batch size.
