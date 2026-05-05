# Phase 14: Causal audit of SAE features (FM/LLM alignment)

## Why this phase exists

Phase 13 produced descriptive labels for every SAE feature by
correlating activations with motif / atom-count / temperature / phase
on the labelling set. These labels say which attribute *co-occurs*
with each feature -- they do not say whether the feature has any
*causal* role in FM2's downstream prediction (per-atom energy).

Without that distinction, when we feed Qwen the labelled top-k
features as evidence (Phase 13 ``full_sae``), the LLM may be reasoning
over decorative directions in FM2's hidden state -- correlated with
labels in the training distribution but causally inert in the head.
That is a misalignment between FM and LLM: the LLM treats the labels
as facts the FM "uses," when in fact the FM may not use them at all.

Phase 14 closes that gap by making each feature label answer to a
counterfactual:

> If we knock feature ``i`` out of the SAE latent before decoding back
> to the CLS, how much does FM2's predicted energy change?

Features whose interventions move the prediction are the ones the FM
is mechanistically using. Those are the features the LLM should hear
about.

## What this phase ships

### `src/fmllm/representation/causal.py`

* ``Intervention`` dataclass (``KNOCK_OUT`` / ``KNOCK_IN`` / ``CLAMP``)
  applied to the SAE latent ``z``.
* ``cls_through_sae`` -- encode -> intervene -> decode; the
  no-intervention path matches the SAE's own reconstruction.
* ``audit_feature`` -- for one feature index, runs the three forward
  paths (original, recon, intervened) on the same audit batch and
  returns a :class:`CausalEffect` record with mean energy per path,
  signed effect = intervened - recon, and ``effect_norm`` =
  ``|effect|`` divided by the inter-specimen energy std (per-feature
  signal-to-noise).
* ``filter_features_by_causal_effect`` -- returns the subset of
  feature indices that pass a configurable norm-effect threshold and
  an activation-rate gate (drops dead features).

### `scripts/sae_causal_audit.py + .sh`

CLI that loads the latest SAE, the latest labels, and the latest
FM2 checkpoint, forwards a 2K-specimen audit set through FM2 once,
and runs ``audit_feature`` for every feature. Output:

```
runs/sae_causal/<run_id>/causal_effects.yaml   # per-feature record
runs/sae_causal/<run_id>/causal_filter.json    # passing feature ids
runs/sae_causal/<run_id>/manifest.yaml
```

### Phase 14.B integration: `full_sae_causal` baseline

`scripts/run_baseline_full_probes.py` accepts a new
``--causal-filter-path`` flag. When set, the SAE prompt slot is
restricted to feature indices in ``passing_feature_ids``. Output is
routed to ``runs/holdout/full_sae_causal/`` so the side-by-side
evaluator picks it up as a new column alongside ``full_sae``.

## Tests (`tests/test_sae_causal.py`)

Eleven CPU tests covering:

* Knock-out zeros only the target column; knock-in / clamp set the
  configured value; the original tensor is not mutated.
* ``cls_through_sae`` with no intervention matches direct ``sae(x)``.
* ``audit_feature`` returns a ``CausalEffect`` with all fields
  populated and consistent under an identity-SAE construction.
* **Discriminative test**: with a constructed SAE+head where the head
  reads only feature ``target``, the audit assigns large
  ``knock_out_effect_norm`` to ``target`` and essentially zero to a
  different feature index. This is the bar the audit has to clear.
* ``filter_features_by_causal_effect`` respects both the norm-effect
  threshold and the activation-rate gate.
* ``normalize_cls`` round-trips through ``denormalize_cls``.
* ``predict_energy`` flattens trailing singleton dim from the head.

## Reproduction

### Local

```
git pull
uv run pytest tests/test_sae_causal.py -v
```

### Remote

Stage 0 (audit):

```
bash scripts/sae_causal_audit.sh                                  # ~5-10 min
```

Stage 1 (the new baseline column):

```
SAE_DIR=$(ls -td checkpoints/sae/*/ | head -1) \
SAE_LABELS_PATH=$(ls -td runs/sae_labels/*/labels.json | head -1) \
CAUSAL_FILTER_PATH=$(ls -td runs/sae_causal/*/causal_filter.json | head -1) \
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_full_probes.sh

BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
```

The evaluator's first line should now read seven or eight columns,
including ``full_sae_causal``.

## What this phase will and will not prove

Will prove:
- Whether FM2's downstream head is mechanistically using the SAE
  features the labels point at, or whether the labels are
  correlation-only summaries of decorative directions.
- Whether restricting the LLM's SAE prompt slot to causally-validated
  features changes goal accuracy (signed comparison vs ``full_sae``).

Will not prove:
- Whether the LLM mechanistically uses the features it sees -- that's
  the next axis (Phase 15 would train an SAE on Qwen's residual
  stream and run analogous interventions on the LLM side).
- Whether causal filtering generalizes off the training distribution.
  All causal scores are computed on training-split specimens; the
  held-out set is used only for the downstream goal-accuracy
  comparison.

## Where this fits

Phase 13 added the ``representation -> text-label`` edge between FM2
and the LLM. Phase 14 adds the ``representation -> causal-effect``
edge so the labels we ship to the LLM are validated on FM2's
mechanistic use, not just on attribute correlations. The intent is a
better-aligned FM/LLM interface: when Qwen sees ``"motif=ring": 2.13``
in its prompt, that string is now a hook on FM2's actual computation,
not an ungrounded summary statistic.
