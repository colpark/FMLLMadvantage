# Phase 10: Self-supervised FM2 (Layer D)

## Why this phase exists

Phase 9 closed with a converging negative result: the supervised FM2
backbone holds selective task-extra signal (probes: phase 0.88 acc,
geometric ~0.65 r²), and a Q-Former connector trained with Stage 1
LM-loss alignment cannot transfer that signal to the LLM in a
specimen-faithful way (inspector: 6 of 8 generations collapsed to a
marginal-mode prior). The architectural reading was that the
representation ceiling, not the connector, was the limiting factor.

Phase 10 tests the alternative hypothesis: that a representation
shaped by predicting RDF *structure* (rather than one scalar derived
from it) carries more task-extra information. The change is upstream
of Phase 9: instead of training a richer connector on a fixed
backbone, we train a richer backbone on the same input data with a
self-supervised objective.

## What I built

### `fmllm.fms.fm2_rdf_ssl`

A parallel FM2 backbone with the same architecture as the supervised
FM2 plus two additions:

- **`mask_token`**: a learnable embedding that replaces masked bin
  embeddings during the SSL forward pass.
- **`recon_head`**: a per-bin MLP that predicts masked bin values
  from the encoder output.

The `encode(rdf)` method returns the unmasked
`(B, rdf_bins + 1, embed_dim)` sequence, *bit-for-bit shape-
compatible* with `fmllm.fms.fm2_rdf.FM2RDFTransformer.encode`. That
guarantee is what lets the existing probing study and Q-Former
connector consume the SSL backbone with a single `--use-ssl` flag.

### `scripts/train_fm2_ssl.py` + `.sh`

The SSL pretrainer. Loads RDFs from `train_50k`, randomly masks 30%
of bins per row (configurable), forwards through the SSL backbone,
and computes MSE loss only on the masked positions. AdamW with
β = (0.9, 0.95), bf16 autocast on CUDA, gradient clipping at 1.0,
20 epochs default.

Output:

```
checkpoints/fm2_rdf_ssl/<train_split>/<run_id>/model.pt
checkpoints/fm2_rdf_ssl/<train_split>/<run_id>/manifest.yaml
checkpoints/fm2_rdf_ssl/<train_split>/<run_id>/training.yaml
```

Saved with the same payload shape `fmllm.fms.common.save_checkpoint`
uses, so `load_checkpoint` works unchanged.

### Probe and connector extensions

`scripts/run_fm2_probes.py` gains a `--use-ssl` flag (and the bash
wrapper a `USE_SSL=1` env var). When set, the probing study loads
the latest checkpoint from `checkpoints/fm2_rdf_ssl/` instead of
`checkpoints/fm2_rdf/` and uses `build_fm2_ssl_model` to instantiate
the architecture. Run id slug includes `ssl` so SSL and supervised
probing reports never collide.

`scripts/train_fm2_connector.py` gains a parallel `--use-ssl` flag.
The connector is trained on top of the SSL backbone if the flag is
set; the saved `connector.pt` records `fm2_kind` so
`scripts/inspect_connector.py` can rebuild the right backbone at
inference time.

### Tests (`tests/test_fm2_ssl.py`)

Six new CPU tests:

- `encode()` returns `(B, 201, embed_dim)` shape.
- `forward(rdf, mask)` returns `(B, rdf_bins)` reconstruction.
- Fully masked inputs produce identical outputs (the mask token is
  the only signal).
- Masked-loss backprop produces non-zero gradient on `mask_token`
  and `recon_head` (sanity check).
- `encode()` is mask-independent.
- `encode()` and `forward()` reject mismatched shapes.

## What the user runs

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest tests/test_fm2_ssl.py tests/test_connectors.py -v
```

Expect 6 + 12 = 18 tests passing.

### Remote 4xH100 host

```
ssh remote
cd ~/FMLLMadvantage
git pull && uv sync --extra dev

# 1. SSL pretraining (~30-60 minutes for 20 epochs on train_50k)
bash scripts/train_fm2_ssl.sh

# 2. Probe the SSL backbone and compare to the supervised one
USE_SSL=1 bash scripts/run_fm2_probes.sh
# Then re-run the supervised probe for an apples-to-apples comparison
bash scripts/run_fm2_probes.sh

# 3. If SSL probes look meaningfully richer (especially n_atoms r2),
#    train a new connector on the SSL backbone
USE_SSL=1 bash scripts/train_fm2_connector.sh

# 4. Inspect side-by-side
bash scripts/inspect_connector.sh -n 8
# (the inspector auto-detects which backbone the latest connector
# was trained on and reports it in the header)
```

### Decision rule

The same rule from Phase 9.0 applies, with the added comparison
against the supervised baseline:

| SSL probe outcome | Decision |
|---|---|
| Probes lift dramatically (e.g. n_atoms r² 0.20 → ≥ 0.7) | Layer D worked. Train connector and re-run inspector; expect specimen-faithful generations. |
| Probes lift moderately (n_atoms 0.20 → 0.4-0.6) | Layer D added information. Connector might still be limited; depends on whether identification-relevant signals (motif, T) are stronger. |
| Probes barely change | Masked-RDF reconstruction did not produce a richer representation. Try contrastive training or move on. |

## What this phase does and does not aim to prove

Aims to prove (or refute):

- Whether a self-supervised objective on the same RDF inputs
  produces a representation with strictly more probe-recoverable
  task-extra signal than the energy-supervised representation.
- If yes: whether a Stage 1 connector trained on this richer
  representation generates specimen-faithful descriptions where the
  Phase 9 connector did not.

Does not prove:

- Whether the representation gain (if any) translates to a
  goal-accuracy improvement under the held-out audit. That is Phase
  10.B (integrate the SSL connector into the OHVD loop) and Phase
  10.C (re-run `scripts/run_holdout.sh` with the SSL connector
  active). Both are conditional on Phase 10 producing a positive
  signal.
- Whether contrastive or cross-modal SSL would do better than
  masked modeling. Single-objective comparison only.

## Where this fits

Phase 9 isolated the architectural question to "is the bottleneck
the connector or the representation?" by showing the connector
trained on the supervised backbone fails. Phase 10 directly probes
the second half: change the representation, keep everything else
the same, see if the Layer C path opens up. If it does, the
research story is "representation matters; SSL pretraining is the
right move for LLM-FM composition on typed-but-bottlenecked
backbones." If it doesn't, the story tightens: even with a richer
representation, Stage 1 alignment alone can't transfer specimen
identity, and the answer is Stage 2 task tuning or a different
connector recipe entirely.

## Phase 10 empirical result (negative, with a twist)

The SSL probing run produced a counter-intuitive headline: every
structural probe got *worse* under masked-RDF pretraining than under
supervised energy regression.

| Probe | Supervised (Phase 9) | SSL (Phase 10) | Δ |
|---|---|---|---|
| n_atoms r² | 0.20 | 0.12 | −0.08 |
| diameter r² | 0.66 | 0.59 | −0.07 |
| mean_coordination r² | 0.63 | 0.47 | −0.16 |
| phase acc | 0.88 | 0.81 | −0.07 |

Three plausible mechanisms drive this direction:

1. **Energy supervision is implicitly extracting structural
   features.** Per-atom potential energy is a lossy function of
   atom count, motif, phase, and coordination, so the backbone
   needs to encode these as intermediates to predict it. The Phase
   9 probes told us those intermediates were partially recoverable;
   the Phase 10 SSL result says the supervised objective is the
   thing that put them there.

2. **Masked-RDF is too local.** The g(r) is smooth and locally
   correlated. Predicting a masked bin reduces to interpolation
   from neighbors. The model can minimize the SSL loss without
   learning global structural features, so the representation
   ends up biased toward local fluctuations rather than the global
   identity the probes test.

3. **20 epochs of pretraining is short, but the gap is wider than
   what training-time alone is likely to close.** Going from 0.12
   to 0.20+ probably requires a different objective, not just more
   steps of masked reconstruction.

### Combined Phase 9 + Phase 10 architectural verdict

- **Phase 9 (Layer C)**: connector on supervised FM2 cannot transfer
  specimen identity to the LLM via Stage 1 alignment.
- **Phase 10 (Layer D, masked-RDF)**: replacing the supervised
  objective with masked-RDF reconstruction *degrades* the
  structural representation on every probe.

The honest read is that, on this testbed, **the supervised energy-
regression objective happens to be a strong structural-representation
learner as a byproduct**, and the path to richer LLM-FM coupling is
not upstream of the connector. Two natural directions remain
unexplored:

- **Stage 2 task tuning** — LoRA the orchestrator LLM end-to-end on
  the actual identification objective with the Phase 9 connector
  rather than on the templated-text alignment surrogate. Can fix the
  template-collapse failure mode of Phase 9 without changing the FM.
- **Contrastive or cross-modal SSL** — replace masked reconstruction
  with an objective that explicitly rewards specimen-distinguishing
  features (SimCLR pairs of perturbed RDFs, or CLIP-style alignment
  between (RDF, image) pairs). Would test whether the Phase 10
  result is a property of *this* SSL objective rather than SSL in
  general.

If neither path is taken, the architectural conclusion stands: on
this testbed, the typed head-output contract is the architectural
ceiling, and the verifier-gated Pipeline A from Phase 8a captures
essentially all the recoverable signal. That is itself a
publishable finding about LLM-FM composition on typed scientific
tasks.
