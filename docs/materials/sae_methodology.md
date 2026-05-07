# SAE methodology — current setup and upgrade path

## What we use today

A **Top-K sparse autoencoder** (Bricken et al. 2023, Gao et al.
2024) on CHGNet's pooled per-structure embedding:

```python
class TopKSAE(nn.Module):
    encoder: Linear(in_dim -> hidden_dim)
    decoder: Linear(hidden_dim -> in_dim, unit-norm columns)
    pre_bias: Parameter(in_dim)

    def encode(x):
        z = relu(encoder(x - pre_bias))
        # Hard Top-K: zero out everything outside the top k per row
        topk_mask = z.topk(k, dim=-1)[1]  # one-hot
        return z * topk_mask
```

Configuration on the materials port:

| Parameter | Value | Justification |
|---|---|---|
| `in_dim` | 64 | CHGNet pooled-embedding dimension |
| `hidden_dim` | 1024 | 16x expansion factor (standard for SAE on small embeddings) |
| `k` | 32 | ~3% activation rate per row |
| `epochs` | 30 | enough on 50K rows for the loss to plateau |
| Optimizer | AdamW(lr=1e-3) | standard for SAE on bounded-magnitude embeddings |

## Why Top-K (not L1)

The classic SAE recipe (Towards Monosemanticity, Cunningham et al.
2023) uses an L1 sparsity penalty: `loss = mse + λ * ||z||_1`. The
problem is `λ` is finicky to tune — too small and the SAE becomes
dense; too large and features die.

Top-K replaces the soft L1 with a *hard* sparsity constraint
(exactly `k` features active per row). No `λ` to tune. Gao et al.
2024 (OpenAI) showed Top-K matches or exceeds L1 SAEs across
architectures and scales while being simpler.

## What `active_frac = 0.0312` means in the training log

`active_frac = k / hidden_dim = 32 / 1024 = 0.03125`. The hard
Top-K constraint forces *exactly* k features active per row, so
this number is constant by construction. The variation comes from
*which* features fire, not *how many*. Don't read the constant
active_frac as a sign of mode collapse.

The metrics that DO indicate health are in
`scripts/materials/06b_diagnose_sae.py`:

- **Dead-feature count**: features that fire on no specimen.
- **Activation-count distribution**: per-feature counts; flat is
  good, long-tailed indicates a few features dominating.
- **Pairwise mask overlap**: how similar the top-k feature sets
  are between random pairs of specimens. Should be near
  `k / hidden_dim` if features are evenly used.
- **Reconstruction-error distribution**: per-specimen MSE; if
  long-tailed, some specimens are poorly covered.

Run `bash scripts/materials/06b_diagnose_sae.sh` after Stage 5 to
see all four.

## The dead-feature failure mode (and remedy)

Without intervention, Top-K SAEs reliably produce **5-15% dead
features** — features that never fire because their encoder
weights settled into a region the input distribution doesn't
reach. Bricken et al. 2023 (Anthropic) introduced **dead-feature
resampling**: every N steps, find features that haven't fired in
the last M steps; reinitialize their decoder columns toward
input-space directions with high reconstruction error; reset
their encoder weights and biases.

This typically drops dead-feature counts to 0-2%. We have not
yet implemented it in this codebase; if the diagnostic script
flags >20% dead features, that's the right next step.

## Sophisticated alternatives (when to upgrade)

| Method | Source | When to upgrade |
|---|---|---|
| **Dead-feature resampling** | Bricken et al. 2023 | Diagnostic shows >20% dead features. Low complexity (~30 LOC), well-validated. |
| **Auxiliary K loss** | Gao et al. 2024 | Diagnostic shows mode collapse (long-tailed activation-count distribution). Adds a small penalty on the top-(k+aux) features beyond the main top-k to keep them productive. |
| **JumpReLU SAE** | Lieberum et al. 2024 (Anthropic) | Want adaptive sparsity (some inputs activate few features, others many). Replaces Top-K with a learnable per-feature threshold. ~50 LOC of new model class. |
| **Gated SAE** | Rajamanoharan et al. 2024 (DeepMind) | Diagnostic shows degraded reconstruction quality after enforcing sparsity. Separates the magnitude prediction from the gating. Better Pareto frontier (sparsity vs reconstruction). |
| **Switch SAE / mixture-of-experts** | various 2024 | Multi-domain data where features for different domains shouldn't share. Probably overkill for our materials port. |

## Decision tree

```
Run scripts/materials/06b_diagnose_sae.sh after Stage 5.

If diagnostic verdict is HEALTHY:
  -> proceed to Stages 6 (label) + 7-9 with current Top-K SAE

If verdict is ISSUES FOUND:
  - dead features > 20%      -> add dead-feature resampling
  - mode collapse            -> add auxiliary K loss
  - per-input variance       -> upgrade to JumpReLU
  - long-tailed recon error  -> larger hidden_dim or longer training
```

## What we expect to see on materials

With 50K specimens and CHGNet's 64-dim pooled embedding:
- **Dead features**: probably <5% (50K specimens cover the embedding manifold well; each of 1024 features has many candidate inputs). LJ Phase 13 had ~95% rare features but on 200 specimens; materials should be much better.
- **Activation-count median**: ~1500 fires per feature (uniform expectation = 50K × 32 / 1024 ≈ 1562). Flat distribution if SAE is healthy.
- **Pairwise mask overlap**: ~0.03 (uniform random expectation). Higher if compositions cluster (likely some clustering since materials are not uniformly distributed).
- **Reconstruction error**: median ~0.005-0.02 in normalized space (training loss converged to 0.011 — RMSE ≈ 0.1, very clean reconstruction).

If the diagnostic confirms these ranges, the current Top-K SAE is fine and we proceed to Stage 7. If it doesn't, we have a clear remediation path.
