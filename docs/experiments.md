# Experimental Program

Year-1 ships five primary experiments. The first three (E1, E2, E3)
make the architectural claim: that bundling FMs through bridges and
a verifier produces something stronger than any FM alone, that
training-time integration beats inference-time integration on OOD
inputs, and that structure-preserving bridges generalize better than
language-anchored bridges. The last two (E4, E5) make the mechanistic
claim: that the verifier's sources contribute non-redundantly, and
that the architecture's value scales with FM quality.

Phase 8 implements the experiment scripts. Phase 9 wires them into
`scripts/reproduce_all.sh`.

## E1, composition curve

**Hypothesis.** Final-prediction accuracy improves as more FMs are
composed, with a saturating-with-overshoot signature past the
optimal subset.

**Manipulation.** Sweep the number of FMs available to the
orchestrator over `{1, 2, 3}`, holding the verifier and bridges
fixed.

**Measurement.** Final-prediction accuracy on held-out specimens.
Per-cell breakdown across the four `(N_axis, T_axis)` cells.

**Pass criterion.** Going from 1 FM to 3 FMs improves accuracy by at
least the verifier's calibrated noise floor on the in-distribution
split, with the OOD splits showing larger gains.

## E2, train-time vs inference-time integration

**Hypothesis.** Pipeline B (RL fine-tuned LLM) outperforms Pipeline
A (off-the-shelf LLM with the same FM tools and verifier), with the
gap widening on OOD inputs.

**Manipulation.** Compare Pipeline A and Pipeline B on
in-distribution and OOD splits.

**Measurement.** Final-prediction accuracy and trajectory-step
recoverability. Reports per-split gap.

**Pass criterion.** Pipeline B beats Pipeline A on OOD by a margin
larger than Pipeline B beats Pipeline A on in-distribution.

## E3, bridge anchor

**Hypothesis.** The structure-preserving bridge generalizes better
than the language-anchored bridge on OOD cluster sizes.

**Manipulation.** Hold the verifier, FMs, and LLM fixed. Swap the
bridge between language-anchored and structure-preserving.

**Measurement.** Final-prediction accuracy on the four OOD cells.

**Pass criterion.** The structure-preserving bridge wins on OOD-N
and OOD-T cells by at least the in-distribution gap between the two
bridges.

## E4, verifier ablation

**Hypothesis.** The five verifier sources contribute non-redundantly.
Disabling individual sources degrades performance, and the gap
widens on OOD inputs.

**Manipulation.** Run the full architecture under five conditions
on the same held-out evaluation set:

- V0, no verification (LLM uses FMs as tools without checks).
- V1, rules only.
- V2, rules + conformal.
- V3, rules + conformal + cross-FM.
- V4, all five sources (rules + literature + cross-FM + simulator + conformal).

The architectural enabler is `Verifier.integrator`, which accepts a
`sources_config` field at runtime. The experiment script overrides
this field per condition.

**Measurement.** Final-prediction accuracy, OOD generalization on
unseen sizes and temperatures, chain-of-thought consistency
(trajectory-step recoverability).

**Pass criterion.** V4 dominates V0 by a margin larger than the
expected per-source contribution. Each successive condition (V1 to
V4) adds at least one statistically distinguishable improvement on
at least one OOD cell.

**Implementation.** `scripts/exp_e4_verifier_ablation.py` (Phase 8).

## E5, FM quality sweep

**Hypothesis.** The verifier's contribution to the architecture scales
with FM quality. As FM quality drops, the verifier's marginal value
grows.

**Manipulation.** Train each FM (FM1, FM2, FM3) at three nested
training-set sizes (`train_10k`, `train_30k`, `train_50k`). Run the
full architecture at each FM-quality level. The split nesting
ensures FM-quality differences come from data quantity not
composition.

The architectural enabler is the `--train-split` flag on
`scripts/train_fm.py`, plus the `train_subsets` block in the splits
YAML.

**Measurement.** Final-prediction accuracy, OOD generalization, and
verifier contribution measured as `accuracy(V4) - accuracy(V0)` at
each FM-quality level.

**Pass criterion.** The verifier contribution at the 10K scale is
at least as large as the verifier contribution at the 50K scale,
ideally larger. A flat verifier contribution curve falsifies the
mechanistic claim.

**Implementation.** `scripts/exp_e5_fm_quality_sweep.py` (Phase 8).

## Architectural vs mechanistic claims

E1 to E3 make the architectural claim:
- E1 confirms composition adds something.
- E2 confirms training-time integration adds more than inference-time.
- E3 confirms structure-preserving bridges generalize better.

E4 and E5 make the mechanistic claim:
- E4 confirms each verifier source contributes non-redundantly.
- E5 confirms the architecture's value scales with FM quality.

The two claims are complementary. The architectural claim says the
bundle is more than the sum of its parts. The mechanistic claim says
specific parts of the bundle do specific work, and we can measure
who does what.

## Quality gates

Each experiment script writes a structured `results.yaml` and a small
PDF report under `runs/<run_id>/report.pdf`. The audit checklist
confirms:

- The script runs end-to-end on a smoke-test subset of the data.
- The results YAML has a stable schema across runs.
- The PDF report includes per-cell metrics and the predicted
  signature against the observed signature.

## Wall-clock budget

| Experiment | Inference cost | Training cost | Total |
|---|---|---|---|
| E1 | ~few hours per architecture | none | ~half day |
| E2 | ~few hours per pipeline | Pipeline B fine-tune (Phase 6) | ~one day |
| E3 | ~few hours per bridge | none | ~half day |
| E4 | ~few hours per condition (5 conditions) | none | ~one day |
| E5 | ~few hours per quality level (3 levels) | 9 FM training runs | ~two days |

The full Phase 9 reproduce-all run takes 2 to 3 days of dedicated
4xH100 time.
