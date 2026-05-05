# Phase 9: Representation connectors (Layer C)

## Why this phase exists

Phase 8a closed with a finding: the LLM consumes only the foundation
models' head outputs (typed scalars), never the backbone
representation. The architecture has FMs in name but uses them as
specialized regressors. Phase 9 tests whether direct access to the
representation adds reasoning capability beyond what the typed-claim
contract delivers.

The pilot is one connector on FM2 only. If it shows measurable gains,
propagate to FM1 and FM3. If it doesn't, the head-output contract was
sufficient and the result reframes the architectural conclusion.

## What I built

### `fmllm.fms.fm2_rdf.FM2RDFTransformer.encode`

A new method that returns the full hidden-state sequence
`(B, 201, 320)` without applying the energy head. `forward` now calls
`encode` and applies the head, so probes and the connector consume
the pre-head sequence while the existing energy path is unchanged.

### `fmllm.connectors`

New subpackage with three modules:

- `qformer.py` &mdash; `FM2Connector`, a Q-Former plus linear
  projection. 32 learnable query tokens cross-attend over FM2's
  hidden states. The output `(B, 32, llm_dim)` is ready to prepend
  into the LLM's input embedding stream. About 8M trainable
  parameters with the default config.
- `text_annotations.py` &mdash; deterministic templated descriptions
  per specimen. Reads `(N, motif, T)` from the HDF5 (and optionally
  equilibrium positions for diameter / coordination). Outputs 1-3
  sentences per specimen.
- `dataset.py` &mdash; (handled inline in the trainer; the
  in-memory `_PairsDataset` caches `(rdf, text)` pairs).

### `scripts/run_fm2_probes.py` (Phase 9.0)

The probing study. Freezes FM2 from the latest checkpoint, extracts
CLS embeddings for a probe split (default `train_50k` capped at 2K
specimens), trains four small probes:

- `n_atoms` (regression, sanity)
- `diameter_lj` (regression, geometric)
- `mean_coordination` (regression, structural)
- `phase` (3-class classification, thermodynamic)

Reports per-probe `R^2` / accuracy and a one-line headline. Decision
rule for whether to commit to Phase 9.A:

| Probe outcome | Decision |
|---|---|
| All probes &ge; 0.85 | Representation is rich. Proceed to connector. |
| Mixed | Selective richness. Build connector, expect modest gains. |
| All near chance | Collapsed to energy. Skip connector; consider self-supervised pretraining. |

### `scripts/train_fm2_connector.py` (Phase 9.A Stage 1)

Stage 1 alignment training. Freezes FM2 and the orchestrator LLM
(default Qwen 2.5 7B Instruct), trains only the Q-Former + projection
with LM loss against the templated annotations.

Layout per row:

```
[connector tokens (Q)] [chat-template prompt (Lp)] [annotation tokens (Lt)]
labels:
[-100 for Q] [-100 for Lp] [annotation tokens for Lt]
```

The connector learns to map FM2's representation into a region of the
LLM's embedding space where the LLM can decode it into specimen-faithful
text. Optimizer: AdamW, default lr 1e-4. Connector trains in bf16 on a
single H100 in roughly an hour for 2K specimens, 3 epochs.

Output:

- `runs/connectors/<run_id>/connector.pt` &mdash; state dict plus the
  fm_dim/llm_dim/n_query metadata needed to rebuild the module at
  inference time.
- `runs/connectors/<run_id>/training.yaml` &mdash; loss history.
- `runs/connectors/<run_id>/manifest.yaml` &mdash; full run config.

## What is *not* yet built

- **OHVD integration.** The connector exists as a trainable module
  and a saved checkpoint; the inference-time path that prepends its
  tokens into the orchestrator LLM's chat is deferred to Phase 9.B,
  pending the probing-study go/no-go.
- **Stage 2 task tuning.** Stage 1 is alignment against templated
  text. Stage 2 (LoRA on the LLM, end-to-end task tuning against the
  N/motif/T identification objective) follows when alignment is
  confirmed to land cleanly.
- **FM1 and FM3 connectors.** Single-FM pilot first.

## Tests (`tests/test_connectors.py`)

Eleven new CPU tests cover:

- `FM2.encode` shape and forward-vs-encode-plus-head equivalence.
- Q-Former forward shape, gradient flow on connector parameters,
  rejection of mismatched `fm_dim`, parameter count in a sane range.
- Templated annotation determinism, content faithfulness (mentions
  N, motif, T), phase thresholds, positions-optional behavior, label
  dict round-trip.

## What the user runs to verify Phase 9

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest tests/test_connectors.py -v
```

Eleven tests; all should pass without GPU.

### Remote 4xH100 host

```
ssh remote
cd ~/FMLLMadvantage
git pull && uv sync --extra dev

# 1. Probing study (Phase 9.0): ~10 minutes
bash scripts/run_fm2_probes.sh

# Read the headline and the report; if all probes are above ~0.85,
# proceed to Stage 1. If mixed, proceed but expect modest gains.
# If all near chance, stop here and revisit FM2's training objective.

# 2. Stage 1 alignment training (Phase 9.A): ~1 hour for default config
bash scripts/train_fm2_connector.sh
```

After Stage 1 lands, the next session wires the connector into the
TransformersLLM inference path so we can re-run the held-out audit
with the connector active. That lives in Phase 9.B.

## What this phase will and will not prove

It will prove:

- Whether FM2's energy-supervised representation holds task-extra
  signal recoverable by linear / 2-layer probes.
- Whether a small Q-Former + projection can align that representation
  with Qwen's input embedding space, measured by templated-text
  reconstruction loss.

It will not yet prove:

- Whether the aligned tokens improve goal accuracy or reduce
  hallucination on the held-out audit. That is Phase 9.B (integration
  into the orchestrator) and Phase 9.C (re-running
  `scripts/run_holdout.sh` with the connector active).

## Where Phase 9 fits

Phase 9 is the architectural-depth follow-up to the Phase 8a audit.
The audit established that the verified composite outperforms its
strawmen on a held-out set. Phase 9 asks the deeper question raised
by that audit: are we using the foundation models or just their
heads, and does it matter? The probing study answers the first half;
the connector evaluation answers the second.
