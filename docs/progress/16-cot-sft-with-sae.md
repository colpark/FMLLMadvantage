# Phase 16: CoT-SFT with SAE-augmented evidence (no verifier)

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

The eight-phase architectural picture so far:

```
naked         0.000   <- no FM, no LLM-tool-use
cot_sft       0.467   <- LLM trained on probes-only CoT
no_verifier   0.540   <- LLM with bridged FM tool messages, no verifier
full_probes   0.562   <- LLM + probes + verifier
full_sae      0.585   <- LLM + SAE labels + verifier
full          0.695   <- LLM + bridged FM tool messages + verifier (ceiling)
```

`probe_head` and `cot_sft_sae` slot in as new columns. The
expected ordering, given prior phases:

```
naked < cot_sft < cot_sft_sae <= probe_head < no_verifier < full
                                ^ open question
```

The interesting question is whether `cot_sft_sae` sits above or
below `probe_head`. If above, the training-time-rich-evidence
recipe pays off. If below, the FM's own head is the better
predictor on its own representation, even given a richly-trained
LLM reasoner.

## Open follow-ups

1. **Coefficient sweep on `top_k_features`.** Try `K = 4, 8, 16`.
   More features may help (richer evidence) or hurt (more
   anchoring).
2. **Drop probes, keep only SAE features.** Tests whether SAE
   alone is sufficient. The expected outcome is below `cot_sft`,
   but it's the cleanest isolation.
3. **Add the verifier to `cot_sft_sae`.** Phase 13's
   `full_sae = 0.585` already does the no-training-time SAE
   injection with verifier. Combining the SFT'd adapter with
   the verifier loop gives one more cell in the comparison.
