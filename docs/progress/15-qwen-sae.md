# Phase 15: SAE on Qwen's residual stream (Golden Gate Claude analog)

## Why this phase exists

Phases 9-14 all closed as converging negatives along the
*representation-as-prompt-evidence* axis: connector tokens, SSL
backbone, CoT-SFT, probe injection, SAE injection, and (forthcoming)
causally-filtered SAE injection -- each path tries to surface FM2's
hidden state to the LLM as text or soft tokens, and each underperforms
the vanilla typed-output + verifier baseline (``full = 0.695``).

The remaining un-tested axis is the LLM side. Phase 15 mirrors the
recipe of Templeton et al. 2024 / "Golden Gate Claude":

  1. forward representative inputs through Qwen with a hook on a
     middle residual-stream layer,
  2. accumulate residual-stream activations,
  3. train a Top-K sparse autoencoder on those activations,
  4. label features by what they fire on (verdict, correctness,
     specimen attributes),
  5. at inference, hook the same layer and add a multiple of one
     SAE feature's decoder column to the residual stream to steer
     the model.

If the four prompt-side negatives say "richer evidence in the prompt
does not help," activation-side steering is the only remaining
representation-reading pathway. It is also the canonical SAE-in-LLM
use case (Templeton et al., Lindsey et al.), so the result connects
this project to a well-defined external literature.

## What this phase ships (Stage A + Stage B)

### `src/fmllm/representation/llm_sae.py`

Hooking primitives, model-architecture-agnostic:

* ``ActivationHarvester(layer_module)`` -- forward-hook context
  manager that accumulates ``(B, T, hidden_dim)`` residual tensors
  flattened to ``(B*T, hidden_dim)`` on CPU in float32. Cleans up
  on exit. Handles both tuple-output (Llama / Qwen DecoderLayer) and
  bare-tensor outputs.
* ``ActivationSteerer(layer_module, feature_direction, coefficient,
  position_mask=None)`` -- adds ``coefficient * feature_direction``
  to the residual stream at the hooked layer, optionally masked to
  specific token positions. The steering recipe from
  Templeton et al.
* ``resolve_layer_module(model, "model.layers.14")`` -- dotted-path
  walker that handles numeric ``ModuleList`` indices.

### `scripts/harvest_qwen_activations.py + .sh` (Stage A)

Replays a prior baseline ``trajectories.jsonl``. For each trajectory:

  1. reconstructs a minimal chat ``[system, user=query,
     assistant=final_claim_json]``,
  2. tokenizes, forwards through Qwen with a hook on
     ``model.layers.14`` (configurable),
  3. captures the residual-stream activation at the *last* token --
     the closing position of the commit JSON.

Output:

```
runs/qwen_activations/<run_id>/activations.npy   # (N, hidden_dim) fp32
runs/qwen_activations/<run_id>/metadata.yaml     # per-row labels
runs/qwen_activations/<run_id>/manifest.yaml
```

The metadata records ``specimen_id``, ``verdict``, ``is_correct``,
``claim``, ``ground_truth``, and ``n_tokens_in_chat`` per row, so
Stage C labelling has everything it needs.

### `scripts/train_qwen_sae.py + .sh` (Stage B)

Loads the harvested activations matrix, normalizes per feature, and
trains a Top-K SAE (default ``hidden_dim=16384``, ``k=64``, 30
epochs). The output checkpoint:

```
checkpoints/qwen_sae/<run_id>/sae.pt
checkpoints/qwen_sae/<run_id>/training.yaml
checkpoints/qwen_sae/<run_id>/manifest.yaml
```

is the basis for Stage C labelling and Stage D activation steering
(both deferred to follow-up phases).

### Tests (`tests/test_llm_sae.py`)

Twelve CPU tests on stub two-layer models:

* ``ActivationHarvester`` captures ``(B, T, H)`` and flattens to
  ``(B*T, H)`` correctly.
* Hook output handling works for both tuple-out and tensor-out
  layers.
* ``pop()`` clears the buffer; context exit removes the hook.
* ``ActivationSteerer`` adds the right delta in both output regimes
  and respects ``coefficient`` and ``position_mask``.
* ``resolve_layer_module`` walks dotted paths and rejects non-module
  resolutions.

## Reproduction

### Local

```
git pull
uv run pytest tests/test_llm_sae.py -v
```

### Remote

Stage A (harvest, ~10-30 min depending on trajectory count):

```
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/harvest_qwen_activations.sh
```

Stage B (SAE training, ~5-15 min on 200 rows; longer on larger sets):

```
bash scripts/train_qwen_sae.sh
```

Defaults harvest from the latest ``runs/holdout/full/`` (200
specimens). For real SAE training, point the harvester at a larger
training-distribution run via ``TRAJECTORIES=...``. The sample size
of 200 is sufficient to demonstrate the pipeline; for monosemantic
features Templeton et al. used millions of activation rows. Scaling
to ~10K rows (one full run on ~2K training specimens, ~5 commit
positions per chat if we extend the harvester) would be the next
step before steering experiments.

## What this phase will and will not prove

### Stage A + B (this PR)

Will prove:
- The hooking pipeline is correct (CPU tests cover the semantics).
- Qwen residual activations on OHVD-shaped chats can be harvested
  and an SAE trained on them at modest cost.

Will not prove:
- Whether any SAE feature has a meaningful semantic role -- that's
  Stage C labelling.
- Whether steering via these features improves goal accuracy --
  that's Stage D.

### Stage C: correlation labelling (now shipped)

`src/fmllm/representation/llm_labels.py` defines
``label_llm_feature`` plus ``rank_features_for_steering``. The
labelling axis set extends Phase 13's specimen-only set with two
task-side axes:

  * ``verdict`` -- ``pass`` / ``caveat`` / ``fail`` / ``null`` from
    the multi-source verifier. Features that lock on CAVEAT are
    candidates for "amplify when uncertain" steering.
  * ``is_correct`` -- whether the trajectory's final claim matched
    ground truth. Features that lock on ``correct=False`` PASS
    rows are the canonical down-clamp targets.

`scripts/label_qwen_sae_features.py + .sh` consumes the Stage A
metadata.yaml + Stage B sae.pt and emits:

```
runs/qwen_sae_labels/<run_id>/labels.json
runs/qwen_sae_labels/<run_id>/details.yaml
runs/qwen_sae_labels/<run_id>/steering_candidates.yaml
runs/qwen_sae_labels/<run_id>/manifest.yaml
```

The ``steering_candidates.yaml`` partitions the locked features into
three pre-ranked lists (``wrong_pass``, ``wrong_any``, ``caveat``)
that Stage D will draw from.

### Stage D: activation steering baseline (now shipped)

`src/fmllm/representation/steered_llm.py` defines
``SteeredLLMWrapper``: wraps any chat-style LLM with an
``ActivationSteerer`` hook that fires on every forward pass during
generation, adding ``coefficient * decoder_column[fid]`` to the
residual at the configured layer.

`scripts/run_baseline_qwen_steered.py + .sh` runs the full Pipeline
A on held-out specimens with this wrapped LLM. Output is routed to
``runs/holdout/full_steered_<fid>_<coef>/<run_id>/`` so the existing
side-by-side evaluator picks it up as a new column. Multiple
(fid, coef) experiments coexist in distinct directories.

Recommended workflow:

```
# 1. Read steering candidates from Stage C
LATEST=$(ls -td runs/qwen_sae_labels/*/ | head -1)
cat "${LATEST}/steering_candidates.yaml"

# 2. Pick a wrong-PASS feature, ablate it (negative coefficient)
FEATURE_IDX=<fid> COEFFICIENT=-2.0 \
    SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_qwen_steered.sh

# 3. Re-evaluate
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
```

The expected comparison against ``full = 0.695``:

| Hypothesis | Result |
|---|---|
| Ablating a "wrong-PASS" feature reduces hallucination | ``full_steered`` hallucination_rate < ``full`` (0.255) at unchanged commit_rate |
| Amplifying a "calibrated abstention" feature shifts more wrong commits to CAVEAT | ``full_steered`` calibrated_abstention > ``full`` (0.59) and total wrong drops |
| No clear feature exists | ``full_steered`` near ``full``, no improvement |

If Stage D produces a positive result on either axis, Phase 15 is the
first un-negative result on representation-reading on this testbed --
because the intervention is at the *activation* level, not as prompt
evidence. (Stages A/B/C with 200-row data may not be enough to find
the right feature; if all candidates land near ``full``, the signal
is the data-volume diagnostic, not a refutation of the steering
recipe.)

## Where this fits

Four converging negatives (Phases 9, 10, 11, 13) ruled out
representation-as-prompt-evidence. Phase 15 tests
*representation-as-activation-intervention* on the LLM. It is the
last natural axis on which "use richer representation" might still
work, and connects the project to the Templeton et al. SAE-in-LLM
canon. Stages A and B ship the prerequisites; Stages C and D are
the experiments that decide it.
