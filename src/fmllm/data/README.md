# fmllm.data

Synthetic Lennard-Jones dataset generation and loading.

Phase 1 will add:
- `generator.py` - CLI script that produces the 50,000-specimen dataset
  with three observation modalities per specimen.
- `dataset.py` - HDF5-backed PyTorch Dataset and DataLoader classes.
- `splits.py` - held-out partitioning logic with explicit per-split
  ID lists saved as YAML.

Currently empty.
