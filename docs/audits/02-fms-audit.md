# Audit Report, Phase 2

**Audited at:** 2026-05-02T01:15:07Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS

## Summary

Phase 2 implements the three foundation models (FM1 image ViT, FM2 RDF
transformer, FM3 trajectory transformer), per-FM training loops with
mixed precision and physics-constraint loss components, split-conformal
calibration, the unified `scripts/train_fm.py` CLI, and the FM tests.
The default config produces parameter counts that land inside the
spec-specified ranges (FM1 10.71M / 10-30M, FM2 7.57M / 5-15M, FM3
12.78M / 10-25M). Local pytest reports **66 passed** in roughly 1
second. I did not run training locally per the
"`user runs training on the remote`" rule from the modified Phase 2
prompt.

## Detailed checks

### CHECK 2.1, FM directory layout
- **Result:** PASS
- **Evidence:** `src/fmllm/fms/{__init__.py, README.md, common.py}`
  plus per-FM subpackages `fm1_image/`, `fm2_rdf/`, `fm3_traj/`, each
  containing `__init__.py`, `model.py`, `train.py`, `conformal.py`,
  `README.md`. `fms/__init__.py` re-exports the three model classes
  plus their builders.

### CHECK 2.2, FM1 model spec
- **Result:** PASS
- **Evidence:** `FM1ImageViT` accepts 64x64 grayscale input (4D or 3D),
  produces categorical count logits over `{0..max_n_atoms}` plus
  `num_queries` position outputs and a confidence logit per query.
  Conv-based patch embedding with `patch_size=8` gives the
  translation-equivariance bias documented in the module docstring
  and in `fm1_image/README.md`. Default config builds 10.71M
  parameters (target 10-30M).

### CHECK 2.3, FM1 training spec
- **Result:** PASS
- **Evidence:** `compute_fm1_losses` combines count cross-entropy,
  Hungarian-matched L2 position loss, BCE objectness loss, and a soft
  box-constraint loss with `box_half_width_lj` from the config.
  Trainer uses AdamW + linear warmup + cosine decay, optional mixed
  precision (`torch.cuda.amp` autocast + GradScaler), validates every
  epoch, saves the best checkpoint by val total loss, writes a
  manifest YAML with training history.

### CHECK 2.4, FM1 conformal spec
- **Result:** PASS
- **Evidence:** `fm1_image/conformal.py` runs the matched-pair L2 over
  the calibration subset, fits per-alpha thresholds via
  `split_conformal_quantile`, writes `calibration.json` next to the
  checkpoint via `write_conformal_calibration`. The default config
  carries `conformal_alpha_levels: [0.10, 0.20]`, which corresponds to
  90% and 80% prediction intervals.

### CHECK 2.5, FM2 model spec
- **Result:** PASS
- **Evidence:** `FM2RDFTransformer` consumes `(B, rdf_bins)` input,
  uses a CLS token, and projects through an MLP energy head to a
  scalar. Default config builds 7.57M parameters (target 5-15M). The
  module docstring captures permutation invariance (input g(r) is
  permutation invariant by construction) and extensive scaling
  (output is per-atom energy by design).

### CHECK 2.6, FM2 training spec
- **Result:** PASS
- **Evidence:** `compute_fm2_losses` uses Huber loss against per-atom
  potential energy (computed by `per_atom_potential_energy` from the
  ground-truth final positions and atom mask) plus a soft
  non-negativity floor penalty controlled by `nonneg_weight`. Trainer
  uses the same AdamW + warmup-cosine + AMP recipe as FM1.

### CHECK 2.7, FM2 conformal spec
- **Result:** PASS
- **Evidence:** `fm2_rdf/conformal.py` records absolute residuals
  `|E_pred - E_true|` over the calibration subset, fits thresholds
  per alpha, writes `calibration.json` with the standard schema.

### CHECK 2.8, FM3 model spec
- **Result:** PASS
- **Evidence:** `FM3TrajTransformer` consumes
  `(B, T, max_n_atoms, 2)` positions and velocities plus an atom
  mask. Per-frame masked-mean and masked-max pooling over real atoms
  gives permutation invariance (verified by
  `test_fm3_permutation_invariance`). A learned temporal positional
  embedding plus a CLS token feed a Transformer encoder. The output
  head returns `(alpha, beta) = softplus(raw) + 1e-3`. Default config
  builds 12.78M parameters (target 10-25M).

### CHECK 2.9, FM3 training spec
- **Result:** PASS
- **Evidence:** `compute_fm3_losses` computes the masked NLL of the
  per-(atom, frame) kinetic energies under
  `Gamma(alpha, 1 / beta)` (PyTorch convention: `rate = 1 / scale`)
  plus a soft equipartition penalty `(alpha * beta - mean_KE)^2`
  weighted by `equipartition_weight`. The NLL test
  (`test_fm3_gamma_nll_matches_torch_distribution`) confirms the
  helper agrees with `torch.distributions.Gamma`.

### CHECK 2.10, FM3 conformal spec
- **Result:** PASS
- **Evidence:** `fm3_traj/conformal.py` records per-specimen mean
  NLLs over the calibration subset, fits thresholds per alpha, writes
  `calibration.json`.

### CHECK 2.11, scripts/train_fm.py unified CLI
- **Result:** PASS
- **Evidence:** `scripts/train_fm.py` exposes `--fm {fm1, fm2, fm3}`,
  `--config`, `--h5-path`, `--splits-path`, `--out-dir`, `--device`,
  `--epochs`, `--calibrate-only`, `--checkpoint`. The dispatcher
  imports and calls the per-FM `train` or `calibrate` function.
  Help output renders cleanly.

### CHECK 2.12, FM tests pass
- **Result:** PASS
- **Evidence:** `pytest -m "not gpu" -v` reports
  `66 passed in 0.95s`. New tests cover forward-pass shapes, the
  Hungarian matcher, FM1 box-constraint loss sign, FM2 non-negativity
  loss only firing below floor, FM3 atom-permutation invariance, the
  Gamma NLL helper agreement with `torch.distributions.Gamma`,
  conformal-quantile and calibration-file round-trip.

### CHECK 2.13, config schema extension
- **Result:** PASS
- **Evidence:** `src/fmllm/utils/config.py` defines `FM1Config`,
  `FM2Config`, `FM3Config` inheriting from `FMConfig` and adding
  per-architecture hyperparameters. The schema rejects unknown keys
  (`extra="forbid"`). `configs/default.yaml` matches the schema and
  validates via `test_load_config_repo_default`.

### CHECK 2.14, parameter counts inside the target ranges
- **Result:** PASS
- **Evidence:** Built each FM at default config and counted trainable
  parameters: FM1 = 10.71M (10 to 30 target), FM2 = 7.57M (5 to 15
  target), FM3 = 12.78M (10 to 25 target). Bumped FM1
  `encoder_depth` from 6 to 8 and FM3 `embed_dim` from 256 to 320
  during the audit so all three land in spec.

### CHECK 2.15, no training run locally
- **Result:** PASS
- **Evidence:** `git status` shows only source, config, doc, test
  changes. No checkpoint files anywhere (`checkpoints/` is empty
  apart from its README). No HDF5 dataset locally.

### CHECK 2.16, manifests written next to checkpoints
- **Result:** PASS
- **Evidence:** Each per-FM trainer calls `write_manifest` on the
  output directory after training completes. The manifest records
  the resolved `FMxConfig`, the dataset paths, the run ID, the
  device, the parameter count, the best validation metric, the per-
  split sizes, and the per-epoch history. Conformal calibrators
  write `calibration.json` next to the checkpoint.

### CHECK 2.17, prose style
- **Result:** PASS
- **Evidence:** Scanned every modified or new markdown file for
  em-dashes and semicolons in narrative prose (excluding fenced code
  blocks). Zero matches.

### CHECK 2.18, working tree clean after Phase 2 commit
- **Result:** PASS (after the Phase 2 commit lands)
- **Evidence:** Pre-commit `git status --short` shows only the
  Phase 2 changes plus the audit doc.

## Fixes applied during audit

- Bumped `FM1Config.encoder_depth` from 6 to 8 (10.71M params, was
  9.14M) so the parameter count lands inside the 10-30M target.
- Bumped `FM3Config.embed_dim` from 256 to 320 (12.78M params, was
  8.19M) so the parameter count lands inside the 10-25M target.
- Mirrored both changes in `configs/default.yaml`.
- Re-ran the full local test suite and confirmed 66 passed.

## Remaining concerns

- **Mixed precision is on by default.** If the remote PyTorch build
  has any AMP-related flakiness, set `mixed_precision: false` per FM
  in the config to disable autocast and the gradient scaler.
- **The Hungarian matcher runs on CPU per-batch.** For batch sizes
  above roughly 256 specimens the matcher latency starts to
  dominate. Profile on the remote and reduce `batch_size` for FM1
  if needed.
- **The empirical Gamma fit on FM3 has a tail risk.** When
  `velocities` contain zeros (rare but possible during the snippet),
  the NLL term clamps via `nll_clip = 50.0`. Adjust the clip if
  losses oscillate.
- **Conformal calibration on the calibration subset.** The default
  `calib_fraction = 0.10` carves 10% of the train pool. With ~40K
  train, that gives ~4000 calibration specimens for FM2 and FM3 and
  ~4000 * mean_N matched pairs for FM1. That is plenty for stable
  finite-sample quantiles.
- **No layer-1 evaluation yet.** The `world-model` style metrics
  (compression, distinction, recoverability) live in Phase 7 and run
  on saved trajectories. The training-time metrics here are
  proximate proxies for those.

## Sign-off

The Phase 2 implementation matches the original prompt's
specification (modified by the local-Claude/remote-execution split).
The user can proceed to running the smoke-train and full training
recipes in `docs/progress/02-fms.md`, then the conformal calibration
step.
