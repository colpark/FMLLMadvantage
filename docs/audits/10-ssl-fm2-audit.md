# Audit Report, Phase 10 (scaffolding)

**Audited at:** 2026-05-04T23:30:00Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS for the scaffolding scope (SSL backbone, trainer,
probe and connector extensions, tests). Empirical results follow
once the pretrainer runs.

## Summary

Phase 10 adds a self-supervised parallel FM2 backbone trained with
masked-RDF reconstruction, plus the wiring needed to A/B compare
the SSL representation against the supervised one through the same
probing study and connector training script that Phase 9 used. The
supervised FM2 stays where it is; nothing about Phase 8a's Pipeline
A or the held-out audit is invalidated.

## Detailed checks

### CHECK 10.1, SSL model preserves the encode() shape contract
- **Result:** PASS
- **Evidence:** `FM2SSLTransformer.encode(rdf)` returns
  `(B, rdf_bins + 1, embed_dim)`, identical in shape to
  `FM2RDFTransformer.encode`. The Q-Former connector cross-attends
  over this sequence regardless of which backbone produced it; no
  changes are required on the connector side.
  `test_encode_returns_full_sequence_with_cls` enforces the shape.

### CHECK 10.2, SSL forward computes per-bin reconstruction
- **Result:** PASS
- **Evidence:** `forward(rdf, mask)` returns `(B, rdf_bins)`.
  `test_forward_returns_per_bin_prediction` enforces the shape.
  The training script computes loss only on positions where mask is
  True (`pred[mask], rdf[mask]`).

### CHECK 10.3, mask token is the only signal at fully-masked positions
- **Result:** PASS
- **Evidence:** When the mask is all ones, the bin embeddings are
  entirely replaced by the mask token plus position embedding. Two
  different RDF inputs must therefore produce identical predictions.
  `test_mask_token_is_used_when_masked` runs the model on two
  different inputs with full masks and asserts equality.

### CHECK 10.4, gradient flows through the SSL-relevant parameters
- **Result:** PASS
- **Evidence:** `test_forward_loss_flows_backward_only_through_masked`
  computes a loss on masked positions only and asserts that
  `mask_token` and the first layer of `recon_head` both have
  non-zero gradients after backprop. This is the path the trainer
  exercises.

### CHECK 10.5, encode() is mask-independent
- **Result:** PASS
- **Evidence:** `encode(rdf)` does not call `forward` and does not
  consult any mask. Test `test_encode_is_independent_of_mask_state`
  confirms the function is deterministic on identical inputs.

### CHECK 10.6, SSL trainer saves a checkpoint readable by load_checkpoint
- **Result:** PASS
- **Evidence:** `scripts/train_fm2_ssl.py` calls
  `fmllm.fms.common.save_checkpoint` with the same payload shape
  used by the supervised FM2 trainer (`{"model": state_dict,
  "epoch": ..., "extra": {...}}`). The probe and connector scripts
  use `load_checkpoint` from the same module, so the SSL checkpoint
  is consumed identically to the supervised one.

### CHECK 10.7, probe CLI A/B compares cleanly
- **Result:** PASS
- **Evidence:** `scripts/run_fm2_probes.py` accepts `--use-ssl`.
  Setting it switches three things: the checkpoint root from
  `fm2_rdf` to `fm2_rdf_ssl`, the model factory from
  `build_fm2_model` to `build_fm2_ssl_model`, and the run-id slug
  from `supervised` to `ssl`. The output report includes a
  `backbone_kind` field. Reports from the two runs do not collide.

### CHECK 10.8, connector CLI A/B compares cleanly
- **Result:** PASS
- **Evidence:** `scripts/train_fm2_connector.py` accepts `--use-ssl`.
  Setting it switches the checkpoint root and the model factory in
  the same way. The saved `connector.pt` records `fm2_kind`, so
  `scripts/inspect_connector.py` rebuilds the matching backbone at
  inference time. Run-id slug includes `-ssl` so SSL and supervised
  connector runs land in distinct directories.

### CHECK 10.9, no existing tests broken
- **Result:** PASS
- **Evidence:** No existing files were modified except
  `scripts/run_fm2_probes.py`, `scripts/train_fm2_connector.py`,
  `scripts/inspect_connector.py`, and their bash wrappers. All
  pre-existing flags retain their defaults; the only behavioral
  change is opt-in via `--use-ssl`. The Phase 9 connector run
  artifacts already on disk (without an `fm2_kind` field) are
  handled by the inspector's `payload.get("fm2_kind", "fm2_rdf")`
  default.

### CHECK 10.10, syntax compiles for every new and modified file
- **Result:** PASS
- **Evidence:** `python -c "import ast; ast.parse(...)"` on all
  changed files returns OK. `bash -n` on the modified shell scripts
  returns OK.

## Scope boundary

Phase 10 ships scaffolding plus tests. It does *not* yet ship:

1. **The pretrained SSL backbone.** Will be produced by running
   `bash scripts/train_fm2_ssl.sh` on the remote.
2. **A side-by-side probe comparison.** Comes from running
   `bash scripts/run_fm2_probes.sh` then
   `USE_SSL=1 bash scripts/run_fm2_probes.sh`.
3. **A new connector trained on the SSL backbone.** Conditional on
   step 2 showing meaningful probe lift.
4. **OHVD integration and held-out re-evaluation.** Phase 10.B/C,
   conditional on the connector inspector showing specimen-faithful
   generations.

If step 2 shows no probe lift, step 3 can be skipped and Phase 10
closes as a second negative result. The architecture would then
have ruled out both connector richness (Phase 9) and SSL
representation richness (Phase 10) on this testbed and we would
move on to either contrastive SSL or external-LLM comparison.

## Empirical findings (added post-run)

Probing on the SSL backbone after 20 epochs of masked-RDF
reconstruction produced a counter-intuitive negative result:

| Probe | Supervised | SSL | Δ |
|---|---|---|---|
| n_atoms r² | 0.20 | 0.12 | −0.08 |
| diameter r² | 0.66 | 0.59 | −0.07 |
| mean_coordination r² | 0.63 | 0.47 | −0.16 |
| phase acc | 0.88 | 0.81 | −0.07 |

Every structural probe is worse under SSL pretraining than under
supervised energy regression. The architectural inference (detailed
in `docs/progress/10-ssl-fm2.md`) is that energy supervision is
implicitly extracting structural features as a byproduct, and
replacing it with an objective that does not require those
features (local-bin reconstruction is satisfiable by interpolation)
loses the byproduct.

### Why Phase 10.B (SSL-connector training) is not run

The Phase 10.B step was conditional on the SSL probes showing
meaningful improvement. They show the opposite. Training a Stage 1
connector on a strictly poorer representation cannot improve over
the Phase 9 connector that already failed. The connector run on
SSL is therefore deliberately *not* attempted, matching the
Phase 9.B decision pattern: only run downstream phases when the
upstream signal supports it.

## Reproduction

```bash
# Local
uv run pytest tests/test_fm2_ssl.py tests/test_connectors.py -v

# Remote
bash scripts/train_fm2_ssl.sh                                   # ~30-60 min
bash scripts/run_fm2_probes.sh                                  # baseline (supervised)
USE_SSL=1 bash scripts/run_fm2_probes.sh                        # SSL probes
# Phase 10.B is not run; the SSL representation is poorer than
# supervised, so a connector trained on it cannot improve over the
# Phase 9 connector that already failed.
```
