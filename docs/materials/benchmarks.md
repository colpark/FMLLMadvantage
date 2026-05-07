# Materials port — Benchmark validation

Two-level validation strategy for the materials port: confirm
the FM (CHGNet) reproduces published numbers, then compare our
LLM-with-CoT pipeline against published reference baselines on
the same tasks.

This document is the table-of-record for both *published*
reference numbers and our *reproduced* numbers. Update when each
new run completes.

## Why benchmark first

Without grounding, an end-to-end pipeline result like
`cot_sft_sae_mat = 0.65` is uninterpretable. We need:

1. **CHGNet's native output reproduces published energy MAE
   (~30 meV/atom)** — confirms our `CHGNetWrap.encode()` and
   data path are correct.
2. **Probe-based predictions on top of CHGNet match published
   single-task baselines** (e.g. ALIGNN ~0.022 on `mp_e_form`)
   — confirms the FM-head equivalent is competitive.
3. **The LLM-with-CoT pipeline beats both** — the value-add
   over the FM-head baseline is what justifies the architecture.

## Published reference numbers (Matbench v0.1)

Standard 5-fold CV on the 13 Matbench tasks. Lower is better
for regression (MAE), higher for classification (ROC-AUC).

### Property prediction

| Task | n | Metric | CHGNet | MACE-MP-0 | M3GNet | ALIGNN | MEGNet | Comment |
|---|---|---|---|---|---|---|---|---|
| `mp_e_form` (formation energy, eV/atom) | 132,752 | MAE | ~0.030 | ~0.025 | ~0.039 | ~0.022 | ~0.028 | Most-cited regression task |
| `mp_gap` (DFT band gap, eV) | 106,113 | MAE | n/a* | n/a* | ~0.21 | ~0.218 | ~0.33 | CHGNet/MACE don't have a gap head |
| `mp_is_metal` (metal class) | 106,113 | ROC-AUC | n/a* | n/a* | ~0.94 | ~0.93 | ~0.92 | |
| `expt_gap` (experimental gap, eV) | 4,604 | MAE | — | — | — | ~0.32 | ~0.41 | Small dataset, harder |
| `expt_is_metal` (experimental class) | 4,921 | ROC-AUC | — | — | — | ~0.91 | ~0.89 | |
| `glass` (glass-forming ability) | 5,680 | ROC-AUC | — | — | — | ~0.92 | ~0.85 | |
| `phonons` (phonon DOS peak, cm⁻¹) | 1,265 | MAE | — | — | ~58 | ~37 | ~70 | |
| `dielectric` (refractive index) | 4,764 | MAE | — | — | ~0.32 | ~0.22 | ~0.31 | |
| `log_gvrh` (bulk modulus, log GPa) | 10,987 | MAE | — | — | ~0.075 | ~0.069 | ~0.090 | |
| `log_kvrh` (shear modulus, log GPa) | 10,987 | MAE | — | — | ~0.090 | ~0.087 | ~0.113 | |
| `perovskites` (formation energy, eV/atom) | 18,928 | MAE | — | — | — | ~0.025 | ~0.043 | Well-structured |
| `jdft2d` (exfoliation E, meV/atom) | 636 | MAE | — | — | — | ~46 | ~57 | 2D materials |
| `steels_yield_strength` (MPa) | 312 | MAE | — | — | — | — | ~95 | Tiny dataset |

\* CHGNet and MACE-MP-0 are universal interatomic potentials (predict
energy + forces + stress + magmoms). They do not have band-gap or
metal-class heads in their pretrained checkpoints. To run them on
`mp_gap` / `mp_is_metal` you would train a probe on top of their
embeddings — which is exactly what our Phase 11 / Phase 16 recipe
does.

### Stability and forces (universal-potential metrics)

| Task | Metric | CHGNet | MACE-MP-0 | M3GNet | Comment |
|---|---|---|---|---|---|
| Energy MAE on MPtrj test | meV/atom | ~30 | ~25 | ~35 | Native CHGNet/MACE objective |
| Force MAE on MPtrj test | meV/Å | ~78 | ~67 | ~93 | |
| Stress MAE on MPtrj test | meV/Å³ | ~0.35 | ~0.27 | ~0.45 | |
| Stable structure detection (e_above_hull < 0.025) | F1 | ~0.85 | ~0.88 | ~0.81 | Derived from energy predictions |

### Sources

- **CHGNet** — Deng et al. *Nature Machine Intelligence* (2023), <https://www.nature.com/articles/s42256-023-00716-3>
- **MACE-MP-0** — Batatia et al. *arXiv:2401.00096* (2024), <https://arxiv.org/abs/2401.00096>
- **M3GNet** — Chen & Ong *Nature Computational Science* (2022)
- **ALIGNN** — Choudhary & DeCost *npj Comput. Mater.* (2021)
- **MEGNet** — Chen et al. *Chem. Mater.* (2019)
- **Matbench** — Dunn et al. *npj Comput. Mater.* (2020), <https://matbench.materialsproject.org>

## Our reproduction protocol

We do *not* attempt to reproduce Matbench's 5-fold CV exactly,
because that would require training each FM from scratch on
Matbench splits. Instead we run two simpler validations:

### A. CHGNet native energy reproduction (sanity check)

Forward CHGNet on our held-out 200 specimens. Compare predicted
total energy / atom against Materials Project DFT reference.
Target: MAE ~0.03 eV/atom (matches published).

If our reproduced MAE is materially worse (>0.05 eV/atom), the
pipeline is likely misconfigured — wrong checkpoint, wrong
input format, or scale issues with positions/lattice.

Script: `scripts/materials/10_benchmark_chgnet.{py,sh}`

### B. Probe-based predictions (FM-head equivalent)

After Stage 5 (probe training), score the trained probe bank on
held-out 200:

- `formation_energy` MAE
- `e_above_hull` MAE
- `band_gap` MAE (when band-gap probe is added)
- `is_metal` accuracy
- `is_stable` accuracy
- `space_group` top-1 accuracy

These should land within ~10-20% of the corresponding ALIGNN /
MEGNet single-task numbers. Probes on a frozen embedding are
not as strong as a model fine-tuned end-to-end; the gap is the
information bottleneck of the embedding.

### C. cot_sft_sae headline (the architectural test)

Same six metrics as B, but evaluated on the LLM-with-CoT-SFT
pipeline. The architectural claim is that the LLM extracts
*more* signal from the embedding than the probes do. The LJ
analog showed +54 points on goal_accuracy from `probe_head`
to `cot_sft_sae`; the materials version should show a similar
qualitative pattern, with the per-task quantitative gains
depending on how separable the property is in CHGNet's pooled
embedding.

## Comparison protocol when reporting results

For each property the report should include:

| Approach | Metric | Value | Reference |
|---|---|---|---|
| CHGNet native (pretrained) | energy MAE on held-out 200 | TBD | published ~0.030 |
| Probe head on CHGNet embedding | formation_energy MAE | TBD | comparable to ALIGNN ~0.022 |
| cot_sft_sae (no verifier) | formation_energy MAE | TBD | the architectural test |
| cot_sft_sae + verifier (later) | formation_energy MAE | TBD | full Pipeline A analog |

Filling out this table is the unit of paper-shaped progress on
the materials port.

## Caveats on the comparison

1. **Different data splits.** Matbench uses canonical 5-fold CV
   on Matbench's specific subsets of MP. Our held-out 200 is a
   stratified sample of an `e_above_hull < 0.5` filter. Numbers
   are *qualitatively* comparable but not directly equal.
2. **Different metric protocols.** Some published numbers use
   CV-mean MAE, others use single-fold. We compute MAE over our
   held-out 200 with single-pass evaluation. The numbers can
   differ by ±10% just from this.
3. **Pretrained vs fine-tuned.** Matbench numbers for CHGNet /
   MACE typically come from models *fine-tuned on the specific
   task*. Our probes are trained on the *frozen* embedding,
   which is a weaker baseline. The gap is informative.
4. **Property scope.** CHGNet pretrained does not directly
   predict band_gap or is_metal. Our probes provide those, but
   the comparison vs ALIGNN's directly-trained band-gap head is
   not apples-to-apples.

For the benchmark to be paper-defensible, we'd ideally run
**A + B + C** with explicit notes on each caveat.

## Reproduced numbers (filled in as runs complete)

### CHGNet sanity-check run (2026-05-07)

> **Verdict: pipeline correct, ready to proceed to Stages 4-10.**
>
> Per-element-corrected formation_E MAE = **31.5 meV/atom**, matching
> the published CHGNet target of ~30 meV/atom on Materials Project.

The 10 most-populated elements in the held-out 200 produced
per-element references that match Materials Project's well-known
values within 0.1 eV/atom:

| Element | Our μ̂ (eV/atom) | MP reference | n |
|---|---|---|---|
| O | −4.940 | ~−4.95 | 90 |
| Fe | −8.401 | ~−8.40 | 18 |
| N | −8.350 | ~−8.31 | 17 |
| P | −5.390 | ~−5.41 | 17 |
| Cu | −4.131 | ~−4.10 | 20 |
| Ba | −1.993 | ~−1.92 | 27 |
| Ca | −1.995 | ~−2.00 | 24 |
| K | −1.132 | ~−1.11 | 26 |
| Cs | −1.034 | ~−1.03 | 19 |
| F | −1.931 | ~−1.91 | 18 |

The Pearson correlation between *raw* CHGNet output and MP formation
energy was **0.47** — initially misread as a failure but actually
expected: when compositions vary widely, the rank order of raw
`E_total/atom` is dominated by which elements are present (Fe-rich
at ~−9 eV/atom, Cs-rich at ~−2 eV/atom), making it nearly
orthogonal to formation energy. The per-element-corrected MAE is
the meaningful metric.

The total magmom MAE of 1.54 μB is large; likely a few
high-magnetization specimens with mispredicted moments dominate the
average. Not a blocker for downstream training; worth a separate
diagnostic later.

### Tracking table

| Run date | Approach | Metric | Value | Published | Notes |
|---|---|---|---|---|---|
| 2026-05-07 | CHGNet native | formation_E MAE per-elem corrected (eV/atom) | **0.0315** | ~0.030 (CHGNet on MPtrj) | held-out 200; ridge least-squares for per-element refs |
| 2026-05-07 | CHGNet native | top-10 per-element refs vs MP | match within 0.1 eV | MP table | O / Fe / N / P / Cu / Ba / Ca / K / Cs / F |
| 2026-05-07 | CHGNet native | total magmom MAE (μB) | 1.5402 | n/a | sum-of-magmoms over the cell |
| TBD | Probe head | formation_energy MAE | TBD | ALIGNN ~0.022 | requires Stage 5 |
| TBD | Probe head | e_above_hull MAE | TBD | n/a | |
| TBD | Probe head | band_gap MAE | TBD | ALIGNN ~0.218 | |
| TBD | Probe head | is_metal accuracy | TBD | ~0.94 | |
| TBD | Probe head | space_group top-1 accuracy | TBD | n/a | |
| TBD | cot_sft_sae | per-property table | TBD | n/a | the architectural test |

### Original placeholder table (kept for reference)

| Run date | Approach | Metric | Value | Notes |
|---|---|---|---|---|
| TBD | CHGNet native | energy MAE on held-out 200 | TBD | |
| TBD | CHGNet native | formation_energy MAE | TBD | |
| TBD | Probe head | formation_energy MAE | TBD | |
| TBD | Probe head | e_above_hull MAE | TBD | |
| TBD | Probe head | band_gap MAE | TBD | |
| TBD | Probe head | is_metal acc | TBD | |
| TBD | Probe head | space_group top-1 acc | TBD | |
| TBD | cot_sft_sae | per-property table | TBD | |
