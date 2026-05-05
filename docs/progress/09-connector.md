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

## What this phase proved (negative result)

Three diagnostics converged on a clear conclusion: **Layer C with a
Stage 1 alignment recipe (frozen FM2, frozen Qwen, LM loss against
templated text) does not transfer FM2's representation into a form
the LLM can use.** The architectural premise that representation
access would let the LLM beat the head-output contract did not
materialize on this testbed.

### Diagnostic 1: probing study

Headline numbers from `scripts/run_fm2_probes.sh` on 2000 frozen-
backbone CLS embeddings:

| Probe | Score | Read |
|---|---|---|
| `n_atoms` | r² = 0.20 | Atom count discarded by FM2's per-atom training objective |
| `diameter_lj` | r² = 0.66 | Geometric scale partially recoverable |
| `mean_coordination` | r² = 0.63 | Local density partially recoverable |
| `phase` | acc = 0.88 | Thermal regime strongly encoded |

Interpretation: selective richness, with `n_atoms` essentially
absent. This bounded the architectural ceiling for any connector
trained on the same backbone.

### Diagnostic 2: feature-shuffle ablation

`scripts/train_fm2_connector.sh` rerun with FM features permuted
within each batch:

| Step | Real features | Shuffled features | Gap |
|---|---|---|---|
| 1 | 5.03 | 5.03 | 0.00 |
| 250 | 0.25 | 0.33 | 0.08 |
| 500 | 0.19 | 0.26 | 0.07 |
| **740** | **0.14** | **0.19** | **0.05** |

The 0.05-nat gap corresponds to roughly 1.5 nats of total information
transferred via the connector across a 30-token answer span. That is
about one categorical's worth of signal (e.g., phase). It is real
but small.

### Diagnostic 3: generation inspection

`scripts/inspect_connector.sh` on 8 specimens. The real-FM
generations collapse to a small set of stereotyped templated
descriptions:

- 6 of 8 specimens produced "13-atom triangular disk" regardless
  of true N (range 9-19).
- All 8 produced "T = 0.24 LJ, solid-like regime" regardless of true
  T (range 0.17-1.86).
- All 8 produced "triangular disk" including specimen 4 whose
  ground-truth motif was "ring".

The zero-FM generations differ qualitatively (no template structure;
hallucinated "hexagonal dimer", units not in the testbed), so the
connector did learn template structure. But it did not learn
specimen identity. The LLM sees the marginal-mode specimen, not the
input.

### Combined verdict

Stage 1 with templated text + frozen LLM is *not* the right
configuration for transferring FM representation to the LLM on this
testbed. The connector ends up encoding a fixed prior over the
dataset's marginal distribution, with very weak per-specimen
conditioning.

## Why this happened

Two compounding reasons:

1. **The text supervision did not require specimen-specific
   conditioning to minimize loss.** Templated descriptions have ~5-6
   varying parts (N, motif, T, phase, diameter, coordination), but
   most of the LM-loss reduction comes from learning the template
   structure itself. Once the connector + frozen Qwen reproduce the
   format, the marginal-mode prediction wins on average.

2. **The frozen Qwen cannot route the connector tokens against an
   end-task signal.** The LM-loss objective rewards reconstructing
   text, not making correct decisions. Without the LLM updating its
   attention patterns to actually use the connector tokens, the
   tokens drift toward whatever pattern minimizes text loss rather
   than what carries useful information.

## What we did *not* try (and where the architectural ceiling
might still be)

- **Stage 2 task tuning.** Add a LoRA adapter on Qwen and train
  end-to-end on the actual identification objective with the
  connector + LoRA optimized jointly. The reward becomes goal
  accuracy, not text reconstruction. Plausibly fixes both failure
  modes above. Not run in this phase.
- **Layer D self-supervised pretraining of FM2.** Replace the
  energy-only supervised objective with masked-RDF modeling or
  contrastive on cluster pairs. The probing study told us the
  ceiling for a frozen energy-supervised backbone is selective
  richness; Layer D raises that ceiling. Out of scope for Phase 9.
- **Larger Stage 1 training.** 2000 specimens × 3 epochs is small,
  but the loss curve flattened by step 500, so more steps probably
  do not change the qualitative picture. Untested.

## Phase 9 outcome and recommendation

The phase shipped scaffolding (probes, connector module, training,
inspection) plus three diagnostics that jointly establish a
publishable negative result. Phase 9.B (OHVD integration) and Phase
9.C (held-out re-evaluation) are *not run*: integrating a connector
that produces marginal-mode generations would not move metrics, and
the inspector confirms this without needing the held-out audit.

Recommended next moves, in priority order:

1. **Close Phase 9 with the negative finding.** The architectural
   conclusion stands: on this testbed, the typed head-output
   contract captures essentially all the recoverable signal. Layer C
   with frozen-frozen LM-aligned connectors does not add value.
2. **Stage 2 task tuning as Phase 9b.** ~1 week of work. Worth doing
   if the architectural question is "does end-to-end task tuning
   change the picture?". Higher-effort, more-informative.
3. **Layer D self-supervised pretraining as Phase 10.** Multi-week
   work, would require regenerating FM2 with a new objective. The
   only path to a higher representation ceiling on this testbed.

## Reproduction

```bash
bash scripts/run_fm2_probes.sh                                  # probes
bash scripts/train_fm2_connector.sh                             # real run
SHUFFLE_FEATURES=1 bash scripts/train_fm2_connector.sh          # shuffle ablation
bash scripts/inspect_connector.sh -n 8                          # inspector
```

## Where Phase 9 fits

Phase 9 is the architectural-depth follow-up to the Phase 8a audit.
The audit established that the verified composite outperforms its
strawmen on a held-out set. Phase 9 asked the deeper question: are
we using the foundation models or just their heads, and does it
matter? The probing study answered the first half (the
representation has selective signal). The connector experiment
answered the second (a frozen-frozen LM-aligned connector does not
make that signal usable to the LLM). The architectural conclusion is
therefore that Phase 8a's typed-claim contract is not artificially
weak, and the next leverage point is upstream of the connector
architecture.
