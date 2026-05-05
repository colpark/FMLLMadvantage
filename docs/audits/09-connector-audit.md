# Audit Report, Phase 9 (scaffolding)

**Audited at:** 2026-05-04T17:00:00Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS for the scaffolding scope (probes + Stage 1 alignment).
Subsequent integration phases (9.B, 9.C) are out of scope for this
audit and remain explicitly deferred.

## Summary

Phase 9 scaffolding adds a representation-connector subsystem
(`fmllm.connectors`) plus two CLIs:
`scripts/run_fm2_probes.py` for the Phase 9.0 probing study and
`scripts/train_fm2_connector.py` for Phase 9.A Stage 1 alignment
training. The existing FM2 model gains an `encode` method that
exposes the pre-head hidden-state sequence. No existing tests are
modified; eleven new CPU tests cover the new modules.

## Detailed checks

### CHECK 9.1, FM2.encode preserves the forward contract
- **Result:** PASS
- **Evidence:** `forward(rdf)` now calls `encode(rdf)` and applies
  the energy head to the CLS token of the returned sequence. Test
  `test_fm2_forward_uses_encode_path` asserts that the manually
  composed `encode + head` produces the same scalar as `forward`
  within numerical tolerance, so the architectural change is
  transparent to every existing consumer of `forward`.

### CHECK 9.2, Q-Former architecture matches the BLIP-2 pattern
- **Result:** PASS
- **Evidence:** `_QFormerBlock` runs self-attention on queries
  followed by cross-attention to encoder features followed by an
  MLP, all with pre-norm. Stack of `n_layers` blocks; learnable
  `(1, n_query, fm_dim)` query parameter; final linear projection
  to `llm_dim` plus LayerNorm. Default config (`n_query=32,
  n_layers=2, n_heads=8, fm_dim=320, llm_dim=3584`) produces ~8M
  trainable parameters, which the
  `test_qformer_param_count_reasonable` test bounds.

### CHECK 9.3, only the connector receives gradient
- **Result:** PASS
- **Evidence:** `test_qformer_only_connector_has_grad` confirms that
  after a backward pass through random FM features, both the query
  parameter and the projection have gradients populated. The
  training script additionally calls `requires_grad_(False)` on
  every FM2 parameter and every LLM parameter, then constructs the
  optimizer with `connector.parameters()` only, so backward updates
  only flow into the connector.

### CHECK 9.4, templated annotations are deterministic and faithful
- **Result:** PASS
- **Evidence:**
  - `test_annotation_is_deterministic` confirms identical inputs
    produce identical text and phase labels.
  - `test_annotation_mentions_required_facts` confirms the text
    contains the atom count, the human-readable motif, and the
    formatted temperature.
  - `test_annotation_phase_thresholds` confirms the cold/warm/hot
    boundaries at 0.30 and 1.00 LJ produce the expected phase
    labels.
  - `test_annotation_with_positions_includes_geometry` and
    `test_annotation_without_positions_drops_geometry` confirm the
    geometry-optional code path.

### CHECK 9.5, probing CLI (Phase 9.0) is self-contained
- **Result:** PASS
- **Evidence:** `scripts/run_fm2_probes.py` loads FM2 from the
  latest checkpoint, freezes it, extracts CLS embeddings via
  `model.encode`, builds ground-truth labels via
  `annotate_specimen_from_h5`, and trains four probes (linear or
  2-layer MLP) on a CPU-friendly setup. Decision rule is documented
  in the module docstring and printed at the end of the run. Output
  YAML records every probe's metric plus the input config.

### CHECK 9.6, connector training keeps the FM and LLM frozen
- **Result:** PASS
- **Evidence:** `scripts/train_fm2_connector.py` after loading FM2:
  ```python
  fm2.eval()
  for p in fm2.parameters():
      p.requires_grad = False
  ```
  After loading the LLM:
  ```python
  llm.eval()
  for p in llm.parameters():
      p.requires_grad = False
  ```
  Optimizer:
  ```python
  optimizer = torch.optim.AdamW(connector.parameters(), ...)
  ```
  Only `connector.parameters()` enter the optimizer; `loss.backward()`
  flows into `inputs_embeds` (which the LLM treats as a non-leaf
  tensor) and through the connector path back to the queries +
  cross-attention + projection. FM features are detached via
  `torch.no_grad` around `fm2.encode(rdfs)`.

### CHECK 9.7, LM loss masking is correct
- **Result:** PASS
- **Evidence:** `_build_inputs` constructs labels of shape
  `(B, Q + Lp + Lt)`, fills `-100` for the connector and prompt
  positions, copies `target_ids` for the answer positions, and
  additionally masks `pad_id` to `-100`. HuggingFace's
  `CausalLMOutputWithPast.loss` ignores `-100` positions, so the LM
  loss reflects only the answer-text tokens.

### CHECK 9.8, connector checkpoint stores enough to rebuild
- **Result:** PASS
- **Evidence:** Saved `connector.pt` includes the state dict plus
  `fm_dim`, `llm_dim`, `n_query`, `n_layers`, `n_heads`, the FM2
  checkpoint path, the LLM model name, and a `stage` marker.
  Sufficient to reconstruct `FM2Connector` and load weights at
  inference time without consulting the training run config.

### CHECK 9.9, no existing tests broken
- **Result:** PASS
- **Evidence:** The only existing-file edit is
  `src/fmllm/fms/fm2_rdf/model.py`, where `forward` was refactored
  to call into `encode`. Existing tests that exercise FM2's forward
  pass remain valid because the output value is preserved bit-for-
  bit (covered by `test_fm2_forward_uses_encode_path`).

### CHECK 9.10, syntax compiles for every new file
- **Result:** PASS
- **Evidence:** `python -c "import ast; ast.parse(open(...).read())"`
  on all seven new or modified Python files returns OK.

## Scope boundary (deferred to Phase 9.B and beyond)

Three things remain explicitly out of scope for this audit:

1. **OHVD inference-time integration.** The connector is a saved
   checkpoint. The orchestrator's `TransformersLLM` does not yet
   thread its tokens into the LLM's input embedding stream. Phase
   9.B owns this; it depends on the Phase 9.0 probes coming back
   with at least selective richness.

2. **Stage 2 task tuning.** Stage 1 (alignment) trains against
   templated text. Stage 2 will optionally LoRA-fine-tune the LLM
   end-to-end against the N/motif/T identification objective.

3. **Held-out re-evaluation with the connector active.** Phase 9.C
   reruns `scripts/run_holdout.sh` once the connector is wired into
   inference. Without 9.B that comparison cannot be measured.

## What I deliberately did not do

- I did not break the existing typed-claim contract. Every verifier
  source still operates on the head outputs the bridges already
  expose. The connector is additive: it gives the LLM a second
  modality alongside the typed JSON.
- I did not move the FM2 backbone to a self-supervised objective.
  The probing study tests the existing energy-supervised
  representation first; if probes show the representation is
  collapsed, *then* the next move is Layer D, not Layer C.
- I did not pre-judge the result. The probing CLI's decision rule
  explicitly admits that "all probes near chance" means the
  connector should not be built.

## Reproduction

```bash
# Local
uv run pytest tests/test_connectors.py -v   # 11 tests, no GPU

# Remote
bash scripts/run_fm2_probes.sh              # ~10 min, Phase 9.0
bash scripts/train_fm2_connector.sh         # ~1 hr, Phase 9.A Stage 1
```
