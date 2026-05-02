# fmllm.fms.fm3_traj

FM3: a trajectory transformer that maps the 100-step MD snippet
(positions and velocities) to the moments of a Gamma distribution
fitting the per-atom kinetic-energy distribution.

## Files

- `model.py` - `FM3TrajTransformer` model and `build_fm3_model(cfg)`
  constructor. A small atom MLP encodes ``(x, y, vx, vy)`` per atom
  per frame; masked mean and masked max pooling aggregate per frame
  for permutation invariance over atoms; learned temporal positional
  embeddings plus a CLS token feed a Transformer encoder; a head
  projects the CLS feature to ``(log alpha, log beta)`` followed by
  ``softplus``.
- `train.py` - the training script. Negative log-likelihood of the
  empirical KE distribution under the predicted Gamma plus a soft
  equipartition penalty pulling ``alpha * beta`` toward the empirical
  mean KE.
- `conformal.py` - split-conformal calibration. Per-specimen
  non-conformity score is the per-(atom, frame) NLL.

## Symmetries

- **Permutation invariance** holds by construction. The per-frame
  pooling averages and maxes over real atoms only; both operations
  commute with any permutation. Padded slots get masked out.
- **Equipartition prior** appears as a soft training penalty: with
  ``d = 2`` and unit mass, ``E[KE per atom] = T = alpha * beta``.
