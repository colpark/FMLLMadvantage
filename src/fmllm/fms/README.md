# fmllm.fms

The three foundation models. Each FM lives in its own sub-subpackage
so the model code, training script, and conformal calibration co-evolve.

## Subpackages

- `fm1_image/` - Vision Transformer with a DETR-style set-prediction
  head over 64x64 grayscale images. Predicts atom count plus the set
  of `(x, y)` positions with one confidence logit per query slot.
- `fm2_rdf/` - 1D Transformer over the radial distribution function.
  Predicts coarse-grained energy per atom. Permutation-invariant by
  construction; extensive scaling holds by output design.
- `fm3_traj/` - Trajectory transformer over the 100-step MD snippet.
  Predicts the per-atom kinetic-energy distribution as the moments of
  a `Gamma(alpha, beta)`. Permutation-invariant via masked pooling;
  equipartition prior appears as a soft training penalty.

## Common utilities

- `common.py` hosts the shared dataloader builder, AdamW + warmup +
  cosine schedule, AMP scaffolding, checkpoint I/O, split-conformal
  quantile, and helpers for per-atom potential energy and per-(atom,
  frame) kinetic energies with masking.

## Training

`scripts/train_fm.py` is the unified CLI. Pass `--fm fm1`, `--fm fm2`,
or `--fm fm3` to dispatch to the corresponding training script.
