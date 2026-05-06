# 05 — Architectural Findings

## The two positives

### Positive 1 — multi-source verifier

> A multi-source verifier on typed FM outputs is a robust
> architectural win.

Empirical evidence: `no_verifier = 0.540` → `full = 0.695`.
+15.5 points lift on the held-out range. The verifier ablations
(V0..V4) confirmed each source contributes; the largest
single contributor was the literature source after the
finite-temperature configuration fix (`compare_energy=False`).

The mechanism is **independent verification of the LLM's
typed output against multiple oracles**. Importantly:

- The verifier does not surface FM internal state to the LLM. It
  cross-checks the LLM's commit against rules, literature, a
  simulator, cross-FM agreement, and conformal coverage —
  externally.
- The verifier provides iterative revision: a CAVEAT or FAIL
  causes the LLM to re-deliberate, often producing a corrected
  commit on the next turn.
- The verifier provides calibrated abstention: 102 of 200
  specimens commit as CAVEAT, of which 60% were genuinely wrong
  (the abstention catches them).

This generalizes. Anywhere you have multiple independent ways to
check a typed output, multi-source verification is a robust
architectural primitive.

### Positive 2 — training-time CoT-SFT over rich representation evidence (Phase 16)

> Training a LoRA-tuned LLM on synthetic CoT records that include
> both probe outputs and SAE-derived feature labels produces a
> single-shot, no-verifier predictor that comes within 4.5 points
> of the full verifier-using pipeline.

Empirical evidence: `cot_sft_sae = 0.650`, vs `cot_sft = 0.467`
(+18.3) and `probe_head = 0.110` (+54). Single forward, no
verifier loop, no iterative revision.

The mechanism is **supervised CoT training over typed evidence
that includes auto-discovered representation features**. The LLM
is taught to read both hand-picked probes (Step 1) and labelled
SAE features (Step 1b) as evidence in its CoT, with the final
commit anchored to ground truth. At inference, this produces
specimen-discriminating commits (198 / 200 unique) on the
held-out range.

A diagnostic decomposition shows the failure mode shifts
qualitatively from `full`'s ring-blindness:

| Group | `full` accuracy | `cot_sft_sae` accuracy |
|---|---|---|
| ring, liquid-like | 0% | 33% |
| ring, solid-like | 0% | 43% |
| tri-disk, liquid-like | 31% | 34% |
| tri-disk, solid-like | 88% | 76% |

The CoT-SFT recipe partially repairs the ring failure mode at a
modest cost on the safest group. The wrong-claim sets of `full`
and `cot_sft_sae` are now qualitatively different, suggesting
their union is smaller than either alone — a natural follow-up
is combining the trained adapter with the verifier loop.

This generalizes if the underlying claim holds: when you have an
FM with a rich representation and want an LLM to use it, the
right pathway is **supervised CoT-SFT over labelled features
extracted from the representation**, not in-context prompt
injection of those features.

## Five negatives on inference-time representation reading

The first six representation-pathway experiments tried to
surface FM2's hidden state to the LLM at *inference time* — as
soft connector tokens, as labels in the prompt, as activation
deltas at runtime, or as the output of an SSL-pretrained backbone.
All underperformed the typed-output + verifier baseline.

| Phase | Pathway | When does evidence reach the LLM | Outcome vs `full` |
|---|---|---|---|
| 9 | Connector tokens (FM2 hidden state as soft tokens) | inference | below |
| 10 | SSL-pretrained FM2 (richer representation) | upstream of inference (unsupervised pretraining) | below |
| 11 | CoT-SFT (LLM trained on probe-only CoT) | training, but probes-only | -0.228 (0.467) |
| 13 | SAE feature labels in prompt | inference | -0.110 (0.585) |
| 14 | Causally-filtered SAE labels in prompt | inference | -0.125 (0.570) |
| 15 | SAE-on-Qwen activation steering | inference | -0.035 (0.660) |

The unifying principle that emerged in Phase 16:

> **Inference-time injection of representation features hurts
> (Phases 9, 13, 14, 15). Training-time injection via supervised
> CoT-SFT helps (Phase 16). Phase 11's negative is a
> probes-only special case of the Phase 16 positive — adding SAE
> labels to the same training-time recipe lifts it +18.3 points.**

The distinction between *what the LLM is taught to read* and
*what the LLM is asked to read in-context* is the architectural
load-bearer. SAE features are useful *evidence* — but only when
the LLM is supervised to use them, not when they are injected
into the prompt of a model that wasn't trained on them.

## What ties the negatives together

Across Phases 9, 11, 13, 14, the same effect appears in different
forms:

> The LLM has a fixed reasoning capacity per turn. Whatever
> richer evidence we put into its input, it tends to anchor on
> rather than synthesize. On clear-cut specimens (PASS bucket)
> the anchoring is harmless. On uncertain specimens (CAVEAT
> bucket, where the LLM has to weigh competing evidence) the
> anchoring dominates the typed FM tool messages and lowers
> accuracy.

This shows up specifically in the verdict-stratified breakdown.
Comparing `full = 0.695` to `full_sae = 0.585`:

- PASS-bucket accuracy: 74.5% vs 71.4% (small, 3-point drop)
- CAVEAT-bucket accuracy: 64.7% vs 46.1% (large, 19-point drop)

The 11-point headline gap is essentially all in the CAVEAT bucket.
The "extra evidence in the prompt" specifically hurts the
borderline cases where reasoning matters most.

Phase 15 is the only experiment that intervenes on the *LLM's
own activations* rather than on its prompt. Its negative is
narrower (3.5 points) but it has its own mechanism:
correlation-based SAE labels do not predict the causal role of a
feature, so steering by label produces the wrong intervention.
The Stage D ablation of `fid=6844` increased hallucination
because the feature was a useful tri-disk-solid detector, not a
wrong-commit cause.

## What the negatives are not

The five negatives are **regime-specific**, not architectural
absolutes:

1. **Frozen 7B-class LLM.** Frontier reasoning models with
   stronger in-context synthesis (o1 / Claude Opus / GPT-4-class)
   may not anchor in the same way. Several of these methods (CoT,
   SAE steering) are known to scale very differently above 7B.

2. **Discretized output space.** The task collapses to a 3-tuple
   `(motif, n_atoms, T)`. Once an FM identifies the right motif
   and N, the labels saturate the available information. Richer
   representation has no labels to inform, by construction.

3. **Closed world.** Train and test draw from the same generative
   model. No domain shift, no novel motifs, no adversarial
   inputs. The verifier has perfect-quality literature references.

4. **Small SAE training data.** Phase 13/14 SAE used 20K rows;
   Phase 15 SAE used 200 rows. Templeton et al.'s production
   results required ~1B activation tokens. Below this scale,
   features are dominant directions of small distributions, not
   monosemantic concepts.

5. **Single-token activation capture.** Phase 15 Stage A captured
   one activation per chat (the last token). A multi-token harvest
   would 50-200x the data without needing more trajectories, and
   would likely surface different features.

The architectural conclusion the data supports is:

> **On a discretized-output testbed with a frozen 7B LLM, two
> architectural pathways achieve near-ceiling goal accuracy:
> (a) typed-output + multi-source verifier (`full = 0.695`); and
> (b) supervised CoT-SFT over rich representation evidence
> (probes + SAE features) with no verifier (`cot_sft_sae =
> 0.650`). Inference-time injection of the same representation
> features is uniformly worse.**

Stronger claims require lifting one or more of the five
constraints above and re-running the same axes.

## What the verifier finding generalizes to

Less constrained:

- **Independence of oracles is the load-bearing property.** The
  verifier worked because rule_library, literature, cross_fm,
  simulator, and conformal each have independent failure modes.
  When two of them disagree, the LLM gets a useful CAVEAT.
- **Discrete typed contracts cap information loss but bound
  information gain.** A `BridgedFMOutput` is a JSON dict; it
  cannot convey continuous structure that doesn't fit its
  schema. Tasks where the answer is structurally not a typed
  scalar will need a different contract.
- **Iterative revision is cheap and effective.** The OHVD loop's
  16-step budget gives the verifier room to flag and the LLM
  room to fix. Single-shot architectures lose this.
- **Calibrated abstention is undervalued.** 51% of `full`'s
  commits are CAVEAT, and the calibrated_abstention rate of 0.59
  means the verifier correctly flags ~60% of wrong commits as
  uncertain. This is the architectural feature that lets the
  system be wrong cheaply rather than confidently.

## The interpretive lessons of Phase 15 specifically

These are worth carrying to other projects:

1. **Correlation-based SAE labels are descriptive, not
   interventional.** "Top activators are wrong-PASS commits" does
   not mean "the feature causes wrong-PASS commits." We tested
   this directly: ablating `fid=6844` (whose top activators were
   100% wrong-PASS commits at purity 1.00) made hallucination go
   *up*. The label was about what activates the feature, not
   about its functional role.

2. **Volume effects dominate small-SAE labels.** The seven
   wrong-PASS candidates all locked on
   `motif=triangular_disk + phase=solid-like`. Tri-disk-solid is
   the *safest* group (12% error), not the *worst*. The locks
   appeared because tri-disk-solid is the dominant volume in the
   200-row sample (64 of 98 PASS commits). The SAE found
   "directions in tri-disk-solid space," not "directions toward
   wrong commits."

3. **Validate the candidate set against the ground-truth failure
   distribution before steering.** Our Phase 15 Stage D would have
   spent fewer compute hours if we had run the
   `inspect_qwen_sae_candidates.sh` diagnostic *before* the
   steering experiment, not after. The ground-truth wrong-PASS is
   concentrated on `(ring, *)` and `(tri_disk, liquid-like)`. No
   SAE candidate locked on either; in retrospect the experiment
   was likely to underperform.

## The interpretive lessons of Phase 16 specifically

These are the new transferable claims as of Phase 16:

1. **Training-time supervised CoT over rich evidence is the
   pathway that works.** Phases 13/14 injected the same SAE
   labels at inference time and lost 11-13 points; Phase 16
   trained the LLM to read them via a synthetic CoT chain and
   gained 18 points (over probes-only Phase 11) plus 54 points
   (over the FM head alone). The recipe is: extract labelled
   features from the FM representation, build synthetic
   step-by-step reasoning that explicitly references those
   features as evidence, fine-tune the LLM (LoRA) on that data
   with the final claim anchored to ground truth.

2. **The LLM is not a lossy filter on the FM head.** With
   training-time rich evidence, the LLM is a much *better*
   predictor than the FM's own downstream head — `cot_sft_sae`
   beat `probe_head` by 54 points. This contradicts the
   default assumption that "an LLM reading FM outputs can at
   best recover the FM's predictive capacity." When supervised
   correctly, it can substantially exceed it.

3. **Failure modes can be moved by training-data design.**
   `cot_sft_sae` partially repaired the ring failure mode that
   `full` was 0% on, at a modest cost on tri-disk-solid. The
   choice of which features to surface in the synthetic CoT
   shapes the failure distribution. This is a controllable
   axis worth exploring.

## How this maps to other LLM-FM projects

The transferable claims:

- **Add a verifier first.** It's the highest-ROI architectural
  primitive. Independence of sources matters more than
  sophistication of any one source.
- **Add SAE-augmented CoT-SFT second.** It's the second-highest-
  ROI primitive on this testbed and the cleanest single-shot
  predictor we found (within 4.5 points of the verifier ceiling
  with no inference-time orchestration). The recipe transfers
  to any FM that has a usable representation amenable to SAE
  decomposition.
- **Don't conflate "richer in at inference" with "better in."**
  Adding more evidence to the prompt of a frozen LLM can hurt
  on borderline cases. Test the simpler typed-contract baseline
  before proposing connectors / SAE labels / probe injections.
- **Validate SAE labels causally before steering on them.**
  Templeton's recipe (correlation labelling) is interpretation,
  not intervention. Phase 14's causal-audit pattern (intervene,
  measure outcome) is the right complement.
- **Match SAE training data to the regime.** 200 rows is
  insufficient. If you can't get to 100K+ activations, expect
  small-data SAE failures.
- **The output space matters.** Discretized classification tasks
  saturate. Open-ended generation, anomaly detection,
  interpolation, and hypothesis generation are where richer
  representation pays off — the LLM-FM architectural choices may
  also flip in those regimes.

## What we are *not* claiming

- That `cot_sft_sae` will beat `full` after combining with the
  verifier. We don't yet know the verifier-augmented version
  of `cot_sft_sae`. The two architectural pathways are still
  separate columns; their union is a follow-up experiment.
- That the recipe scales unchanged to frontier LLMs. Single-
  shot CoT-SFT positives at 7B may saturate or amplify with a
  larger reasoning model.
- That the result transfers to continuous / open-ended outputs.
  The discretized `(motif, n_atoms, T)` output space is the
  testbed; tasks where the answer is a generated specimen, an
  anomaly score, or a discovery may behave very differently.
- That the 0.695 ceiling is fundamental. With a better LLM or
  different verifier configuration, both ceiling and floors may
  shift.
- That inference-time SAE steering doesn't work for LLM-
  orchestrated systems generally. We tested one feature at one
  coefficient on one direction in a 200-row-trained SAE. The
  conclusion "Templeton-style steering doesn't help here" is
  not the same as "Templeton-style steering doesn't help."
