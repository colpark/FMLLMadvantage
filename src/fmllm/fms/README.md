# fmllm.fms

Three foundation models trained on different observation modalities.

Phase 2 will add:
- `fm1_image/` - small Vision Transformer for 64x64 grayscale images.
  Predicts atom count and per-atom positions.
- `fm2_rdf/` - 1D Transformer over the radial distribution function.
  Predicts coarse-grained energy per atom.
- `fm3_traj/` - Trajectory Transformer over 100-step MD snippets.
  Predicts kinetic-energy distribution moments.

Each FM keeps `model.py`, `train.py`, and `conformal.py` next to one
another so the training loop and the conformal calibration co-evolve.

Currently empty.
