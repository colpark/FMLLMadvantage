# Audit Report, Phase 13 (scaffolding)

**Audited at:** 2026-05-05T20:00:00Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS for the Stage 0 + Stage 1 + Stage 2 scaffolding.
The held-out evaluation runs in Stage 2; results are not yet
available.

## Summary

Phase 13 ships an auto-discovery layer on top of FM2's
representation: train a Top-K sparse autoencoder, label its
features by attribute correlation, inject the labels into
Pipeline A's user prompt. The goal is to surface representation
content the hand-picked probe bank doesn't pre-specify, in a
textual form the LLM can read.

The phase is fully additive on the existing pipeline. No existing
script or config changes default behavior; SAE injection is opt-in
via `--sae-dir`.

## Detailed checks

### CHECK 13.1, TopKSAE forward shape and sparsity invariants
- **Result:** PASS
- **Evidence:** `tests/test_sae.py::test_topk_sae_forward_shape`
  asserts `recon` and `z` shapes. `test_topk_sae_enforces_sparsity`
  asserts at most `k` nonzero entries per row. The Top-K mask is
  applied via `topk` + `scatter_` rather than a sort, so the
  sparsity is exact regardless of input distribution.

### CHECK 13.2, decoder renormalization is correct
- **Result:** PASS
- **Evidence:** `test_topk_sae_decoder_renorm_keeps_unit_columns`
  perturbs decoder weights, calls `_renormalize_decoder()`, and
  asserts column norms are 1.0 within float tolerance. The
  trainer calls renormalization after every optimizer step.

### CHECK 13.3, label_feature uses correlations correctly
- **Result:** PASS
- **Evidence:** Four label tests cover the categorical lock for
  motif and phase, the continuous descriptor for temperature, and
  the fallback "unlabelled" path when no pattern exists. Each test
  builds a controlled fake-attributes set so the expected behavior
  is unambiguous.

### CHECK 13.4, label rendering is deterministic
- **Result:** PASS
- **Evidence:** `label_feature` only consults summary statistics
  (top-N indices via argsort, Pearson correlation via numpy). All
  inputs are arrays; given the same inputs, the rendered label is
  identical. No nondeterministic ordering.

### CHECK 13.5, SAE checkpoint round-trips
- **Result:** PASS
- **Evidence:** `train_sae.py` saves a payload with state_dict,
  in_dim, hidden_dim, k, cls_mean, cls_std, fm2_checkpoint, and
  training metadata. `label_sae_features.py::_load_sae` rebuilds
  the architecture from those fields and loads weights with
  strict=True. Same loading path is used in
  `run_baseline_full_probes.py::_load_sae_and_labels`.

### CHECK 13.6, integration with Pipeline A is opt-in
- **Result:** PASS
- **Evidence:** Without `--sae-dir`, the runner's behavior is
  unchanged from Phase 12. With `--sae-dir`, the output directory
  switches to `runs/holdout/full_sae/<run_id>/` and SAE-derived
  feature labels appear in the user message. Existing run-id
  directories under `runs/holdout/full_probes/` are not touched.

### CHECK 13.7, evaluate_baselines.sh auto-discovers full_sae
- **Result:** PASS (by inheritance)
- **Evidence:** Phase 8a's evaluate_baselines.sh was updated to
  scan all subdirs under BASELINES_ROOT for `*/trajectories.jsonl`
  rather than using a hardcoded list. `full_sae` is a new subdir;
  it will be picked up automatically when populated.

### CHECK 13.8, syntax compiles for every new file
- **Result:** PASS
- **Evidence:** `python -c "import ast; ast.parse(open(...).read())"`
  on all six new or modified Python files returns OK.
  `bash -n` on all four shell scripts returns OK.

## Scope boundary

Phase 13 ships the scaffolding plus tests. It does *not* yet ship:

1. **The trained SAE itself.** Produced by running
   `bash scripts/train_sae.sh` on the remote.
2. **The labels.json artifact.** Produced by running
   `bash scripts/label_sae_features.sh`.
3. **The held-out comparison.** Produced by running the
   probe-augmented Pipeline A with `SAE_DIR=...` set, then the
   evaluator.
4. **A shuffle ablation for SAE labels** (randomly permute labels
   across specimens to confirm the LLM is using the labels). Would
   be a useful diagnostic but is not strictly required for the
   first measurement.
5. **LLM-assisted feature labelling** (use a frontier LLM to
   describe what each feature's top activators have in common,
   beyond the categorical/continuous taxonomy this phase uses). A
   higher-fidelity alternative; not built.

If Phase 13 lifts the held-out goal accuracy, items 4 and 5 become
the natural follow-ups. If it doesn't, the architectural read
sharpens: hand-picked probes already span the text-readable
content of the FM2 representation, and the only unmeasured lever
is continuous-token coupling.

## Reproduction

```bash
# Local
uv run pytest tests/test_sae.py -v   # 8 tests, no GPU

# Remote
bash scripts/train_sae.sh                                       # Stage 0, ~15-30 min
bash scripts/label_sae_features.sh                              # Stage 1, ~5-10 min
SAE_DIR=$(ls -td checkpoints/sae/*/ | head -1) \
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_full_probes.sh                    # Stage 2
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh  # comparison
```
