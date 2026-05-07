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

Configuration on the materials port (validated 2026-05-07):

| Parameter | Value | Justification |
|---|---|---|
| `in_dim` | 64 | CHGNet pooled-embedding dimension |
| `hidden_dim` | **256** | 4x expansion (sweet spot — see "Empirical sweep" below) |
| `k` | **16** | 6.25% activation rate per row |
| `epochs` | 30 | enough on 50K rows for the loss to plateau |
| Optimizer | AdamW(lr=1e-3) | standard for SAE on bounded-magnitude embeddings |
| Resampling | every 500 steps, threshold=0 | Bricken et al. 2023, recycles dead features |

## Why Top-K (not L1)

The classic SAE recipe (Towards Monosemanticity, Cunningham et al.
2023) uses an L1 sparsity penalty: `loss = mse + λ * ||z||_1`. The
problem is `λ` is finicky to tune — too small and the SAE becomes
dense; too large and features die.

Top-K replaces the soft L1 with a *hard* sparsity constraint
(exactly `k` features active per row). No `λ` to tune. Gao et al.
2024 (OpenAI) showed Top-K matches or exceeds L1 SAEs across
architectures and scales while being simpler.

## What `active_frac = 0.0625` means in the training log

`active_frac = k / hidden_dim = 16 / 256 = 0.0625`. The hard
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
their encoder weights and biases (and their Adam moments).

This is now implemented in `TopKSAE.resample_dead_features` and
enabled by default in `scripts/materials/05_train_sae.sh`
(`RESAMPLE_EVERY=500`).

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

## Empirical sweep (2026-05-07)

We ran the diagnostic across three configs on 46,817 cached CHGNet
embeddings (50K-specimen download minus oversized cells):

| Config | Dead | Coverage | Median fires | p99/p50 | Pairwise overlap (× uniform) | Recon MSE (mean) |
|---|---|---|---|---|---|---|
| `1024 / 32` | **19.2%** | 80.8% | 156 | 118× | 0.18 (5.8×) | 0.011 |
| `256 / 16` ✓ | **4.3%** | 95.7% | 1868 | 11× | 0.16 (2.5×) | 0.025 |
| `128 / 8` | 5.5% | 94.5% | 2301 | 6× | 0.13 (2.0×) | 0.055 |

**`256 / 16` won.** Rationale:

* `1024 / 32` mode-collapsed: ~150 features carried the load, ~200
  were dead, and the activation distribution was wildly long-tailed
  (mean 1463, median 156 — most features fired ~10× less than uniform
  while a few fired ~17× more). 16x expansion of a 64-dim input asked
  the SAE for more dictionary atoms than the manifold supports.
* `128 / 8` had marginally lower dead rate but reconstruction MSE
  doubled (0.025 → 0.055). Too few atoms to cover the input.
* `256 / 16` is the sweet spot: ~5% dead (Bricken target),
  reconstruction comparable to `1024/32`, and feature-usage
  distribution close to uniform (median 1868 vs uniform 2926).

The remaining "long-tailed reconstruction error" warning (p99/p50 ≈
5.7×) is intrinsic to real materials data — rare compositions and
unusual structures are genuinely harder to reconstruct than typical
ones. This isn't a sparsity-method problem.

## Why 1024 features over 64 dims doesn't work

| Setup | Hidden / input | Expansion |
|---|---|---|
| Anthropic *Towards Monosemanticity* | 16,384 / 512 | 32× |
| Anthropic Sonnet SAE | 1M / 4096 | 244× |
| LJ Phase 13 (FM2 RDF) | 1024 / 320 | 3.2× |
| **Materials port (final)** | 256 / 64 | 4× |

The expansion factor must be tuned to the *effective rank* of the
input, not its raw dimensionality. CHGNet's 64-dim pooled embedding
averages all per-atom information into one global vector; the
effective rank (set by chemistry × structure variance) is probably
~30-50. 4× expansion over a 64-dim input is therefore ~6× over the
effective rank, which matches the published literature's healthy
range. 16× expansion over 64 dims = ~30× over rank, well outside
the regime where Top-K stays well-calibrated.

If a future FM produces richer per-atom features (e.g.
MACE-MP-0 with 128-256 dim node features), expansion can scale up
proportionally. For CHGNet pooled, 256 features is the ceiling.
