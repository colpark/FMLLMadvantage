# fmllm.fms.fm3_traj.probes

Behavioral probes for FM3 (trajectory -> Gamma KE moments).

## Files

- `equipartition.py` - confirms ``alpha * beta`` lies within a
  configured fractional band of the empirical mean per-atom kinetic
  energy.
- `distribution_normalization.py` - numerically integrates the
  predicted Gamma density and confirms it lies near unity.
- `distribution_non_negativity.py` - confirms ``alpha`` and ``beta``
  stay strictly positive (the softplus + epsilon parameterization
  guarantees this).
