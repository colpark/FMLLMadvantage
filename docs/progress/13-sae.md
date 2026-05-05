# Phase 13: SAE feature labels for FM2's representation

## Why this phase exists

Phase 11 demonstrated that supervised probes plus inference-time
text reasoning let the LLM read FM2 evidence beyond what the
energy head exposes. But probes only cover what we *pre-specify*
to predict (atom count, motif, phase, coordination, peak position).
Whatever else the FM2 representation encodes -- features the
project never thought to label -- stays invisible.

Phase 13 adds an automatic feature-discovery layer on top of the
representation: train a sparse autoencoder on FM2's CLS embedding,
let the SAE learn what features the rep decomposes into, label
those features by their correlation with dataset attributes, and
inject the most-active labelled features per specimen into the
LLM's prompt alongside the existing PROBES payload.

The architectural goal is the third leg of the
``(input, representation, probe_outputs)`` triangle: probes already
cover the rep -> probe_outputs edge; this phase adds a
**rep -> text-label** edge, with the labels discovered automatically
rather than pre-specified.

## What I built

### Stage 0: Top-K SAE training

`src/fmllm/representation/sae.py` defines `TopKSAE`. Single linear
encoder + Top-K sparsity + single linear decoder. Standard
post-Anthropic recipe: structural sparsity via Top-K (no L1 tuning),
optional decoder unit-norm renormalization after each step,
pre-encoder bias absorbing the dataset mean.

`scripts/train_sae.py + .sh` trains the SAE on frozen FM2 CLS
embeddings drawn from a configurable number of specimens (default
20K from train_50k). Saves to `checkpoints/sae/<run_id>/sae.pt`.
Default config: hidden_dim 1024, k 32, 30 epochs, AdamW lr 1e-3.
Wall clock under 30 minutes on one H100.

### Stage 1: feature labelling by attribute correlation

`src/fmllm/representation/labels.py` contains `label_feature`. For
each SAE feature it identifies the top-N most-activating specimens,
checks whether they concentrate on a single motif or phase
(categorical lock at >= 70% purity), and computes Pearson
correlation of activation with atom count and temperature.
Continuous descriptors fire at |r| >= 0.30. The result is a
`FeatureLabel` with structured fields plus a rendered label string.

`scripts/label_sae_features.py + .sh` runs the labelling over the
full SAE codebook and emits two outputs:

```
runs/sae_labels/<run_id>/labels.json     # feature_idx -> "f142: motif=triangular_disk + T-cold(r=-0.35)"
runs/sae_labels/<run_id>/details.yaml    # full structured FeatureLabel records
```

### Stage 2: inject labels into Pipeline A's prompt

`scripts/run_baseline_full_probes.py` (extended in this phase)
gains three new flags:

* `--sae-dir`: trained SAE directory.
* `--sae-labels-path`: explicit labels.json (default: latest).
* `--sae-top-k-prompt`: how many top-active features to surface
  per specimen (default 8).

When `--sae-dir` is set, the runner forwards each specimen through
FM2 + SAE, picks the top-k active features, looks up their labels,
and embeds them in the user message:

```
Identify the specimen's atom count, motif, and temperature.
Use FM tools to gather evidence, propose a claim, and commit when confident.

PROBES (derived from a frozen FM2 representation, treat as
approximate hints, not ground truth): {n_atoms: ..., motif: ...}

SAE_FEATURES (top-k labelled directions in FM2's representation
that activated for this specimen): {"f142: motif=triangular_disk":
0.92, "f77: T-cold(r=-0.35)": 0.84, ...}
```

Output then writes to `runs/holdout/full_sae/<run_id>/` (vs
`runs/holdout/full_probes/` for the non-SAE configuration), so
`scripts/evaluate_baselines.sh` discovers it as a distinct sixth
baseline column alongside `naked / cot_sft / no_verifier / full /
full_probes / full_sae`.

The same OHVD loop, the same five-source verifier, and the same
calibrated abstention all still apply -- the change is purely
additional evidence in the user message.

### Tests (`tests/test_sae.py`)

Eight CPU tests:

- TopKSAE forward shape, sparsity enforcement (at most k nonzero
  activations per row), decoder renormalization keeping unit-norm
  columns, full-capacity case (k == hidden_dim).
- `label_feature` locking motif when top activators concentrate on
  one motif, locking phase similarly, firing the temperature
  continuous descriptor when activations correlate with T, falling
  back to "unlabelled" when no pattern is present.

## What the user runs to verify Phase 13

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest tests/test_sae.py -v
```

Eight tests; all should pass without GPU.

### Remote 4xH100 host

```
ssh remote
cd ~/FMLLMadvantage
git pull && uv sync --extra dev

# Stage 0: train the SAE (~15-30 minutes)
bash scripts/train_sae.sh

# Stage 1: label features (~5-10 minutes)
bash scripts/label_sae_features.sh

# Stage 2: run probe-augmented + SAE-augmented Pipeline A on the
# locked held-out range
SAE_DIR=$(ls -td checkpoints/sae/*/ | head -1) \
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_full_probes.sh

# Evaluate everything (auto-discovers full_sae as a new column)
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
```

The four-way (or now five/six-way) comparison table will include a
new `full_sae` column. The headline question:

> Does the auto-discovered SAE feature vocabulary lift Pipeline A's
> goal accuracy beyond what hand-picked probes already provide?

Expected outcomes and their interpretation:

| If `full_sae` lands at | Interpretation |
|---|---|
| ~0.70-0.74 (no change vs `full` and `full_probes`) | The probe vocabulary already covers what's recoverable; SAE-discovered features add no new task-relevant signal. |
| ~0.74-0.80 (modest lift over `full_probes`) | SAE features capture structure the probes don't pre-specify. The auto-discovery layer adds value. |
| ~0.80+ | SAE labels meaningfully widen the LLM's view of the representation. Phase 13 succeeds. |

## What this phase will and will not prove

Will prove:
- Whether Top-K SAE training on FM2 CLS yields features that
  correlate cleanly with dataset attributes (categorical-lock
  rate, continuous-descriptor rate, fraction of unlabelled
  features).
- Whether automatically-discovered + labelled features add
  task-relevant signal beyond hand-picked probes when injected
  into Pipeline A's prompt.

Will not prove:
- Whether the LLM is *using* SAE labels vs ignoring them. A
  shuffle ablation (randomly permute feature labels across
  specimens) would isolate this. Not built in this phase.
- Whether continuous-token (Q-Former / cross-attn) coupling
  would do better than text-rendered SAE labels. That's still
  the unmeasured cell of the Phase 9 / Phase 12 framework.

## Where this phase fits

Phase 8a established the typed-output + verifier baseline (0.695).
Phase 9 ruled out richer connectors. Phase 10 ruled out richer
representations via SSL pretraining. Phase 11 measured trained
inference-time CoT-on-probes (0.467 alone). Phase 12 stacked
probes on top of Pipeline A. **Phase 13 is the
representation-discovery layer**: rather than pre-specifying what
to extract from the rep, let an SAE discover features and label
them automatically, then surface those labels as text the LLM can
reason over.

If Phase 13 lifts the held-out goal accuracy, the architectural
conclusion sharpens: *automatic feature discovery on the rep beats
hand-picked probes*. If it doesn't, the conclusion tightens
again: hand-picked probes already span what's recoverable from
the rep through a textual interface, and the only remaining
architectural lever is continuous-token coupling (the unmeasured
LLaVA-style cell of the Phase 9 framework).

Either result is a publishable finding about how to wire FM
representations into LLM reasoning on typed scientific tasks.
