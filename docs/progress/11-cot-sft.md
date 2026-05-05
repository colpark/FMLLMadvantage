# Phase 11: Synthetic-CoT SFT bootstrap (Stage 2)

## Why this phase exists

Phases 9 and 10 closed with converging negative results: connector
training on a frozen LLM does not transfer specimen identity, and
self-supervised pretraining of FM2 produced a poorer representation
than the supervised one. Both negatives shared a structural
limitation: the LLM was frozen, and its training signal (templated-
text alignment) did not reward solving the actual identification
task.

Phase 11 changes both. The LLM is fine-tuned (LoRA), the training
signal is constructed to reward task-correct reasoning over typed
probe outputs, and the FM stays where it is. The architectural
question becomes: when the LLM is trained to read probe-derived
typed facts and reason explicitly about them, does it produce
specimen-faithful reasoning chains the verifier finds correct?

This is the inference-time CoT-over-probes pattern, with the LLM
trained on synthetic supervision rather than the FM trained on a
new objective.

## What I built (Stage 0 prerequisites + Stage 1 + Stage 2)

### Stage 0 -- probe bank

`src/fmllm/training/probe_bank.py` defines `ProbeBank`, a dict of
named probes (regression or classification) with a unified
`evaluate(features)` API and full save/load round-trip. Each probe
is a 1- or 2-layer MLP from FM hidden dim to either a scalar or a
class-logit vector.

`scripts/train_probe_bank.py + .sh` train five probes on top of the
frozen supervised FM2:

| Probe | Kind | Target |
|---|---|---|
| `n_atoms` | regression | atom count from HDF5 |
| `motif` | classification | motif id from HDF5 |
| `phase` | classification | derived from temperature |
| `coordination` | regression | mean first-shell coordination from positions |
| `peak_position` | regression | first peak of the RDF |

Saves to `checkpoints/probes/<run_id>/`.

### Stage 1 -- synthetic CoT generator

`src/fmllm/training/synthetic_cot.py` is the heart of Phase 11. It
defines `generate_cot(probe_outputs, ground_truth)` that emits a
deterministic four-step reasoning chain:

```
Step 1 - Read the probes:
  - atom-count probe: N ≈ 11.2 (confidence 0.86)
  - motif probe     : triangular_disk (confidence 0.79)
  - phase probe     : solid-like (confidence 0.92)
  - coordination    : 3.40 neighbors per atom
  - RDF first peak  : 1.13 LJ units

Step 2 - Cross-check coordination against the structural guess:
  An 11-atom triangular_disk cluster should have mean coordination
  ≈ 3.34. Observed 3.40 (difference 0.06). The probes are consistent
  on structure.

Step 3 - Resolution:
  All probes agree on a coherent structural picture. The phase probe
  disambiguates the temperature regime, and the RDF peak position
  confirms the LJ length scale.

Final commit: {"motif": "triangular_disk", "n_atoms": 11, "temperature": 0.20}
```

Three architectural points matter:

1. **Probes are explicitly named in Step 1.** The LLM learns the
   probes are inputs it should reference, not noise to ignore.
2. **Cross-check uses physics, not probe consensus.** The
   `expected_coordination(N, motif)` function encodes domain
   knowledge: rings have coordination 2.0, linears average
   `2(N-1)/N`, triangular disks grow from ~3.0 at N=5 to ~4.4 at
   N=30. Step 2 checks consistency between the probes and these
   physical expectations.
3. **The final commit comes from ground truth, not from the
   probes.** This teaches the LLM that probes are evidence and
   truth is the output. If the n_atoms probe says 25 but truth says
   11, the rendered CoT references the wrong probe value
   verbatim while still committing 11 -- so the LLM learns to
   handle disagreement rather than blindly follow probes.

`build_sft_record` wraps the CoT in a (system, user, assistant)
chat structure compatible with Phase 6's `train_sft`.

`scripts/build_cot_dataset.py + .sh` iterates over training
specimens, runs FM2, runs the probe bank, calls
`build_sft_record`, and writes one record per line to
`runs/cot_datasets/<run_id>/records.jsonl`.

### Stage 2 -- SFT on synthetic CoTs

`scripts/train_cot_sft.py + .sh` loads the JSONL, calls
`fmllm.training.sft_trainer.train_sft` with the records, and saves
a LoRA adapter under `checkpoints/cot-sft/<run_id>/adapter/`.

Default hyperparameters: 3 epochs, lr 1e-4, LoRA r=16 alpha=32,
gradient accumulation 16, max sequence 2048, bf16. About 2-4 hours
on one H100 for 10K records.

### Tests (`tests/test_synthetic_cot.py`)

Eleven CPU tests:

- ProbeBank evaluate-shape, save-load round trip.
- `expected_coordination` matches physical intuition for ring,
  linear, triangular_disk.
- `coordination_consistent` thresholding works correctly.
- `generate_cot` is deterministic, mentions every probe, commits
  ground truth verbatim, and renders the inconsistent branch when
  the probes disagree.
- `build_sft_record` emits the exact (system, user, assistant)
  chat shape `train_sft` consumes.

## What the user runs to verify Phase 11

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest tests/test_synthetic_cot.py -v
```

Eleven tests; all should pass without GPU.

### Remote 4xH100 host

```
ssh remote
cd ~/FMLLMadvantage
git pull && uv sync --extra dev

# Stage 0: train the probe bank (~5-10 minutes)
bash scripts/train_probe_bank.sh

# Stage 1: emit synthetic CoT records (~5-15 minutes for 10K records)
bash scripts/build_cot_dataset.sh

# Stage 2: SFT on the records (~2-4 hours for 10K records)
bash scripts/train_cot_sft.sh
```

After training, the resulting adapter can be evaluated by passing
it to the existing baseline runner. Two paths:

1. **Wrap the existing OHVD pipeline with the adapter.** Add
   `ADAPTER_PATH=checkpoints/cot-sft/<run_id>/adapter` to the
   environment when calling `bash scripts/run_baseline.sh full`.
   The pipeline runs Pipeline A with the trained adapter on top of
   Qwen.
2. **Build an explicit probe-conditioned baseline.** Phase 11.B
   (not yet built) would produce a baseline runner that prepends
   probe outputs to the user message as expected by the trained
   adapter. This is the more principled comparison.

## What this phase will and will not prove

Will prove (after running Stage 2 + held-out audit):
- Whether SFT on synthetic CoTs that explicitly reference probes
  produces an LLM that reads probe outputs in its reasoning.
- Whether the rule-based CoT structure transfers to free-form LLM
  reasoning at inference time, or whether the LLM simply
  reproduces the templates verbatim.

Will not yet prove:
- Whether the trained behavior generalizes beyond the training
  distribution (probes generalize independently of the LLM).
- Whether STaR-style rejection-sampling (Stage 3) or GRPO (Stage 4)
  on top of Stage 2 adds further gains. Both are unbuilt and
  conditional on Stage 2 working.

## Where this fits

Phase 8a is the architectural baseline (verifier-gated typed
output, 0.695 goal accuracy on held-out). Phases 9 and 10 ruled
out two natural extensions (richer connector, richer
representation). Phase 11 is the third axis: train the LLM
itself on the existing typed evidence using a CoT-over-probes
recipe. If Stage 2 lifts the held-out goal accuracy or reduces
hallucination, the architectural conclusion is "LLM training
matters, even when the typed-output contract is doing its job."
If it doesn't, the conclusion tightens further: the typed-output
contract on this testbed is at the architectural ceiling for
LLM-FM composition, and the next research target is a different
testbed.
