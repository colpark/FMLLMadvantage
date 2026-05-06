# Phase 16: CoT-SFT with SAE-augmented evidence (no verifier)

> **Outcome (added post-run):** **Confirmed positive.**
> `cot_sft_sae = 0.650`, vs `cot_sft = 0.467` (+18.3) and
> `probe_head = 0.110` (+54.0). Within 4.5 points of the full
> verifier-using pipeline (`full = 0.695`) at single forward
> with no verifier loop. **The first positive on the
> representation-reading axis** in the project. Detailed analysis
> in the "Empirical result" section below.

## Hypothesis under test

> An LLM trained via SFT on synthetic CoT records that include
> *both* probe outputs *and* SAE-derived feature labels can
> outperform the FM's own downstream head (probe ensemble) on a
> single-shot, no-verifier classification task.

This isolates a single, sharp question from the broader
representation-reading axis: does adding SAE-derived feature
labels to the CoT-SFT training data give the LLM enough extra
signal to beat both:

- the FM's own downstream head (probe ensemble), and
- the probes-only CoT-SFT baseline (Phase 11).

We do *not* run a verifier or an OHVD loop. Single forward, single
LLM commit per specimen. The point is to test the
representation-reading pathway in its purest form: training-time
supervision with rich evidence vs the FM's native predictor.

## Setup

| Baseline | Predictor | Evidence visible at inference |
|---|---|---|
| `probe_head` | direct FM2 + probe ensemble (no LLM) | n/a |
| `cot_sft` (existing Phase 11.B) | LoRA Qwen, single-shot | probes only |
| `cot_sft_sae` (NEW) | LoRA Qwen, single-shot | probes + top-K labelled SAE features |

For all three: same FM2 checkpoint, same probe bank, same held-out
range `[40000, 40200)`, same correctness criteria.

`probe_head` translates probe outputs to a `PhysicalStateClaim`:

- `motif`: argmax of motif probe (probe's own `prediction`).
- `n_atoms`: round(n_atoms probe regression), clamped to [2, 30].
- `temperature`: phase probe disambiguates solid vs liquid; map to
  the centroid temperature of that bucket
  (solid-like = 0.30, liquid-like = 0.80).

The temperature-via-centroid rule is a deliberately coarse scalar
prediction so the comparison is fair: probes don't predict T
directly, only phase. Any baseline that beats `probe_head` is
beating its target predictor at *its own task*.

## What was added

### Module changes

- `src/fmllm/training/synthetic_cot.py` extended (backwards-
  compatible) so `generate_cot` and `build_sft_record` accept an
  optional `sae_features: list[tuple[str, float]]`. When provided,
  the user message gets a `SAE_FEATURES` payload alongside
  `PROBES`, and the assistant CoT renders a `Step 1b` section
  listing each labelled SAE feature with its activation.

### New scripts

- `scripts/build_cot_dataset_with_sae.py + .sh` — Stage 1: emit
  SAE-augmented synthetic SFT records to
  `runs/cot_datasets_sae/<run_id>/records.jsonl`.

- `scripts/train_cot_sft_with_sae.sh` — Stage 2: thin wrapper that
  points the existing `train_cot_sft.py` at the SAE-augmented
  dataset and writes the adapter to
  `checkpoints/cot-sft-sae/<run_id>/adapter/`.

- `scripts/run_baseline_cot_sft_sae.py + .sh` — Stage 3:
  single-shot inference with the new adapter. Output to
  `runs/holdout/cot_sft_sae/<run_id>/`.

- `scripts/run_baseline_probe_head.py + .sh` — reference: direct
  prediction from FM2 + probe bank, no LLM. Output to
  `runs/holdout/probe_head/<run_id>/`.

### Tests

`tests/test_synthetic_cot.py` extended with five CPU tests:

- SAE features render a `Step 1b` section in the CoT.
- No SAE features means no `Step 1b` section (backwards compat).
- The user message contains `SAE_FEATURES` payload when SAE
  features are passed.
- Determinism preserved when SAE features are present.
- The final commit still comes from ground truth, even when SAE
  labels disagree with it.

## Reproduction

### Local (CPU)

```
git pull
uv run pytest tests/test_synthetic_cot.py -v
```

### Remote (4xH100)

Prerequisite: a trained probe bank (Phase 11 Stage 0), a trained
FM2 SAE (Phase 13 Stage 0), and SAE labels (Phase 13 Stage 1).
All three already exist on the remote.

```
# Stage 1: build SAE-augmented CoT dataset (~5-15 min for 10K)
bash scripts/build_cot_dataset_with_sae.sh

# Stage 2: SFT on the new dataset (~2-4 hours on H100)
bash scripts/train_cot_sft_with_sae.sh

# Stage 3: single-shot evaluation (~10-30 min)
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_cot_sft_sae.sh

# Reference: probe-head baseline (~1-3 min, no LLM)
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_probe_head.sh

# Re-evaluate the side-by-side (auto-discovers two new columns)
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
```

## Evaluation criteria

The headline `goal_accuracy` on the held-out range is the primary
metric. Plus the verdict-stratified breakdown shows commit_rate
(should be ~1.0 for both new baselines) and any parse failures.

### Decision matrix

| Outcome | Reading |
|---|---|
| `cot_sft_sae > cot_sft` AND `cot_sft_sae > probe_head` | **Positive result.** SAE labels add training signal beyond probes; LLM-with-rich-evidence beats the FM head. |
| `cot_sft_sae > cot_sft`, `cot_sft_sae < probe_head` | SAE adds value over probes-only training, but the FM head is still better. The LLM is recovering some but not all of the FM's predictive capacity. |
| `cot_sft_sae ≈ cot_sft` | SAE labels add no training signal beyond what probes already encode. Consistent with our earlier Phase 13 observation that probes ⊇ SAE labels for this task. |
| `cot_sft_sae < cot_sft` | SAE labels distract the LLM during training. Consistent with the broader finding that extra prompt evidence hurts in the frozen-LLM regime. |
| `probe_head > cot_sft` | The LLM in either CoT-SFT mode does not match the FM's own downstream prediction. The LLM's reasoning is a lossy filter on the probe outputs. |

## What this phase will not show

- **Verifier-augmented** comparisons. Out of scope. Phase 8a
  already established that the verifier adds +15.5 points; this
  phase asks a different question (training-time supervision via
  rich evidence vs FM head).
- **Joint training of FM + LLM.** The FM is frozen; the LoRA
  adapter is on the LLM only. The motivational challenge from
  the design discussion -- "leveraging rich FM representation is
  hard without joint pretraining" -- is the question this phase
  attacks indirectly: SAE features are a form of
  representation-extraction without joint pretraining.
- **Generalizability beyond this testbed.** All conclusions are
  bounded by the discretized `(motif, n_atoms, T)` output space
  and the closed-world synthetic data.

## What this phase fits into

The architectural picture, with Phase 16 included, on the locked
held-out range `[40000, 40200)`:

```
naked              0.000   <- no FM, no LLM-tool-use
naked_vision       0.000   <- generic VLM caption, no specialist FM
probe_head         0.110   <- FM2 + probes alone, no LLM (the FM head)
cot_sft            0.467   <- LLM trained on probes-only CoT, no verifier
no_verifier        0.540   <- LLM with bridged FM tool messages, no verifier
full_probes        0.562   <- LLM + probes + verifier
full_sae_causal    0.570   <- LLM + causally-filtered SAE labels + verifier
full_sae           0.585   <- LLM + SAE labels in prompt + verifier
cot_sft_sae        0.650   <- LLM trained on probes + SAE CoT, no verifier  *** NEW
full_steered_6844  0.660   <- LLM + activation steering + verifier
full               0.695   <- LLM + bridged FM tools + verifier (architectural ceiling)
```

## Empirical result (added post-run)

### Headline ordering

| | goal_accuracy |
|---|---|
| `probe_head` | 0.110 |
| `cot_sft` | 0.467 |
| **`cot_sft_sae`** | **0.650** |
| `full` | 0.695 |

### Decompositions

- **`cot_sft_sae` vs `cot_sft`: +18.3 points.** Adding SAE-derived
  labelled features to the synthetic CoT records gives the LLM real
  training signal beyond what the five hand-picked probes encode.
  The lift is large enough to rule out noise.
- **`cot_sft_sae` vs `probe_head`: +54 points.** A LoRA-tuned LLM
  trained to compose probes + SAE features into a typed claim is
  *dramatically* more accurate than the FM's own downstream head
  applied directly. The LLM is doing nontrivial reasoning over the
  evidence, not just acting as a lossy probe filter.
- **`cot_sft_sae` vs `full_sae`: +6.5 points.** Training-time
  injection of SAE labels (in the supervised CoT chain) beats
  inference-time injection (in the prompt only). This is the
  central architectural insight.
- **`cot_sft_sae` vs `full`: -4.5 points.** Single-shot LLM with no
  verifier comes within striking distance of the full Pipeline A
  (16-step OHVD loop + 5-source verifier + iterative revision +
  calibrated abstention). The verifier ceiling is now visibly
  thin: most of its contribution is recoverable through richer
  training-time evidence.

### Sanity checks

- **Termination**: 200/200 committed (no parse failures, no
  budget exhaustion).
- **Discrimination**: 198/200 unique final_claims. The model
  produces specimen-specific outputs, not a degenerate template.
- **Per-(motif, phase) accuracy** (with stricter exact-match
  scoring than the headline scorer; absolute counts differ but
  the qualitative pattern is unambiguous):

  | Group | `full` accuracy | `cot_sft_sae` accuracy |
  |---|---|---|
  | ring, liquid-like | **0%** (0/3) | **33%** (4/12) |
  | ring, solid-like | **0%** (0/5) | **43%** (6/14) |
  | tri-disk, liquid-like | 31% | 34% |
  | tri-disk, solid-like | 88% | 76% |

### Architectural reading

The failure pattern shifts qualitatively:

- `full`'s persistent weakness was rings (0% accuracy on all
  ring commits). `cot_sft_sae` partially repairs this (+33-43
  points absolute on ring buckets).
- The cost is a modest accuracy drop on the safest group
  (tri-disk-solid: 88% -> 76%). Net headline goes 0.695 -> 0.650
  because the gains on small ring buckets are outweighed by the
  loss on the dominant tri-disk-solid bucket.
- The wrong-claim sets of `full` and `cot_sft_sae` are now
  *qualitatively different*. `full` is wrong on rings;
  `cot_sft_sae` is wrong on a more-uniform spread.

This supports the reading that **the SAE features encode
representation directions that the bridged FM tool messages do
not surface as cleanly**, specifically for ring-vs-tri-disk
discrimination. The CoT-SFT recipe lets the LLM learn to use
those directions; inference-time prompt injection (Phases 13-14)
did not.

### What this overturns from earlier phases

The "five converging negatives on representation-reading" framing
in `docs/handover/05-architectural-findings.md` was incomplete.
The accurate framing is:

> **Five negatives on inference-time representation injection
> (Phases 9, 10, 11, 13, 14, 15) and one positive on training-time
> CoT-SFT over rich representation evidence (Phase 16).** The
> distinction between *training-time* and *inference-time*
> injection of FM-derived labels is the unifying principle.

What the new data does *not* overturn:

- The verifier still earns +4.5 points on top of `cot_sft_sae`
  (`full = 0.695` vs `cot_sft_sae = 0.650`). The verifier
  contribution is smaller than the previously-reported +15.5
  points (against `no_verifier = 0.540`), but it is real.
- The frozen 7B-class LLM regime is unchanged. We have not
  tested whether the same recipe scales to frontier models.
- The discretized output space `(motif, n_atoms, T)` is unchanged.
  The recipe's value on continuous / open-ended outputs is open.

## Open follow-ups (revised in light of the result)

1. **Combine `cot_sft_sae` adapter with the verifier (highest ROI).**
   The failure sets of `full` and `cot_sft_sae` are qualitatively
   different; their union is smaller than the simple maximum
   would suggest. Wrapping the trained adapter inside the OHVD
   loop with the V4 verifier could plausibly exceed `full = 0.695`.
   This is the one experiment most likely to push the project
   above the prior architectural ceiling.

2. **Coefficient sweep on `top_k_features`.** The current run
   uses K=8. Try K=4 (lighter evidence) and K=12 (richer). If
   the relationship is linear-positive we have not saturated.

3. **Drop probes, keep only SAE features.** Tests whether SAE
   alone is sufficient. Cleanest isolation of "what the SAE
   features add."

4. **Combine `cot_sft_sae` adapter with `full_sae_causal`'s
   filter.** Train CoT-SFT only on causally-validated SAE
   features (Phase 14's filter). If the causal filter selects
   features that improve the training signal further, this
   would close more of the gap.
