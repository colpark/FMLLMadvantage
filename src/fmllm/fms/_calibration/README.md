# fmllm.fms._calibration

Cross-FM agreement calibration. Each FM trains its conformal
calibrator independently. After all three FMs at a given training
scale finish, this subpackage measures pairwise empirical agreement
on shared causal variables and writes a tolerance matrix the
verifier's cross-FM source reads.

## Files

- `cross_fm_tolerance.py` - `compute_cross_fm_tolerances` plus
  `CrossFMToleranceMatrix` and YAML I/O helpers. The verifier
  source `verifier/sources/cross_fm.py` (Phase 4) reads the matrix
  to decide whether to flag a disagreement.
