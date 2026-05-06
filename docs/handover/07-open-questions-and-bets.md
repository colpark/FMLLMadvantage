# 07 — Open Questions and Bets for Next Research

This document is the strategic synthesis: given everything we've
learned, what's the highest-payoff direction for someone picking
this work up?

## Highest-ROI follow-up (read first)

**Combine the Phase 16 `cot_sft_sae` adapter with the Phase 4-8a
verifier.** This is the single most likely path to push the
project's ceiling above `full = 0.695`. The argument:

- `full = 0.695` (Phase 8a, with verifier) is wrong on rings 100%
  of the time and on tri-disk-liquid 69% of the time.
- `cot_sft_sae = 0.650` (Phase 16, no verifier) partially repairs
  the ring failure (33-43% accuracy) at modest cost on tri-disk-
  solid (88% → 76%).
- The two pipelines have qualitatively *different* wrong-claim
  sets. Wrapping the trained adapter inside the OHVD loop and
  giving it the verifier should catch each pipeline's wrongs.

**Cost:** ~30-50 min on remote (one Pipeline A run with the
adapter loaded).

**Predicted ceiling:** plausible to reach 0.72-0.76 if the
failure-set-disjointness argument holds.

The single command (single line, no backslash):

`ADAPTER_PATH=$(ls -td checkpoints/cot-sft-sae/*/adapter | head -1) SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json OUT=runs/holdout_combined nohup bash scripts/run_baseline.sh full > /tmp/full_with_sft_sae.log 2>&1 &`

(Output to a separate `runs/holdout_combined/` so the existing
`full` column is preserved. Then re-evaluate after both finish.)

This is the experiment to run before any of the open questions
below.

## What's open (within the existing testbed)

These are scoped to ~weeks of work on the existing infrastructure.

### Open question A: does CoT survive at scale?

Phase 11 stopped at Stage 2 (synthetic CoT bootstrap). Stages 3-4
(STaR rejection sampling on verifier-PASS trajectories + GRPO with
verifier-shaped reward) are unbuilt. With Phase 16 having shown
that training-time rich evidence carries the LLM to 0.650 single-
shot, it is plausible that Stages 3-4 on the SAE-augmented dataset
would push past `full = 0.695`. They could also close some of the
23-point gap between `cot_sft = 0.467` and `full = 0.695`.

**Cost:** ~1-2 weeks. Phase 6's GRPO trainer exists; the new
piece is the prompt format.

**Predicted ceiling:** plausible to reach 0.7+ when applied on
top of `cot_sft_sae`'s 0.650 base; harder to predict for the
probes-only Phase 11 trajectory.

### Open question B: SAE at scale on Qwen activations

Phase 15 used 200 rows. Templeton et al. used ~1B activations.
Two cheap unlocks before any compute-intensive scaling:

1. **Multi-token harvest.** Modify `harvest_qwen_activations.py`
   to capture all assistant tokens per chat (~50-200 per chat),
   not just the last token. Same 200 trajectories produce
   10K-40K activation rows. ~1 hour to implement.

2. **Larger trajectory source.** Run a `full` Pipeline A on the
   training split (5K+ specimens) and harvest from there. ~1 day
   of GPU.

Combined, this gives 50K-200K activation rows — enough to surface
features the 200-row SAE missed (especially for the
`(ring, *)` and `(tri_disk, liquid-like)` failure modes that
have no candidate in our current SAE).

**Cost:** ~1 week including re-running labelling and steering
sweep.

**Predicted result:** unknown. The most likely outcomes are
either (a) "same negatives, more data didn't change the
conclusion" (would significantly strengthen the negative claim) or
(b) "now there's a feature whose ablation actually helps" (would
flip the Phase 15 result).

### Open question C: coefficient sweep + amplification on Phase 15

Stage D ran one experiment: `fid=6844, coef=-2.0`. The full sweep
sketched but unrun:

```bash
for FID in 6844 11357 16320; do
    for COEF in -4.0 -2.0 -1.0 +1.0 +2.0; do
        FEATURE_IDX=$FID COEFFICIENT=$COEF \
            SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
            bash scripts/run_baseline_qwen_steered.sh
    done
done
```

15 experiments × ~30-50 min = ~10-12 hours of GPU. Especially
informative is **positive** coefficient on `fid=6844` — if the
feature is a useful tri-disk-solid detector (our retrospective
hypothesis), amplifying it should improve performance, not hurt.

**Cost:** ~1 day of GPU.

**Predicted result:** mostly null effect (most coefficients land
near `full = 0.695`). The interesting case is amplification of
`fid=6844`: if it helps, the small-data SAE is finding real
detectors but the labels are misleading; if it doesn't, the
features at this scale are noise.

### Open question D: clean rerun of `full_probes`

The current 0.562 number is over 400 duplicated trajectories. A
clean rerun would give the actual figure. Also worth doing the
trim-and-recompute as a sanity check that the qualitative
finding (probe injection hurts vs `full`) is robust to the
duplication.

**Cost:** ~30-50 min on remote.

## Three bets for next research

These are larger-scope and more strategic. In increasing order
of payoff and risk.

### Bet 1 — Mechanistic interp of LLM-FM coupling at scale

*Templeton on Qwen-with-tools.*

**Claim being tested:** there exists a sparse-feature decomposition
of Qwen's residual stream during tool use that includes
*causally identifiable* features for "trusts this tool message"
vs "doubts this tool message," "FM-grounded" vs "confabulating,"
"will commit" vs "will abstain." Identifying these enables
activation-level intervention that complements the verifier.

**Method:** scale Phase 15 Stage A to 1B+ activations (run
Pipeline A on the full training split with multi-token harvest,
on a frontier model where feasible). Train a properly-sized SAE
(hidden_dim 100K+). Causally validate features via interventions
on goal accuracy directly (not via correlation labelling).

**Why impactful:** the LLM-as-orchestrator paradigm is
mechanistically opaque. Rigorous interpretability of how an LLM
reads tool messages would inform every downstream system: agents,
RAG, scientific assistants, code agents. Modest theoretical risk;
well-trodden methodology.

**Cost:** 1-2 researcher-years, frontier-model API access,
~$10K-100K compute. Realistic only with institutional support.

### Bet 2 — Causal abstraction between FM and LLM

*Formal correspondence between FM features and LLM concepts.*

**Claim being tested:** features in the FM's representation are
causally linked to representational subspaces in the LLM via the
typed bridge. Establishing the causal map (intervening on the
FM's feature `f` → predictable change in the LLM's subspace `S`,
and vice versa) would give a formalism for "what does this FM
tell that LLM about the world."

**Method:** combine Phase 14's FM-causal-audit pattern with
Phase 15's LLM-SAE pattern. For each FM SAE feature, intervene at
the FM, observe the LLM's downstream representation. Build a
many-to-many mapping. Validate by holding out features and
showing predicted LLM-side effects.

**Why impactful:** alignment-relevant. We don't have a formalism
for "FM and LLM share concepts." Causal abstraction would supply
one. Connects to wider mechanistic interpretability literature
(Geiger, Wu, Vig, et al.).

**Cost:** novel methodology required; substantial mathematical
infrastructure. 2-3 researcher-years; high risk; very high payoff
if it works.

### Bet 3 — Reframe the testbed for open-ended discovery

*Tasks where the labels can't carry the load.*

**Claim being tested:** our entire negative-result chain is a
consequence of a discretized output space. Tasks structurally
incompatible with typed-tuple output force richer representation
to do real work. Re-running the same architectural axes there
will produce different conclusions.

**Method:** redesign the testbed around tasks where the answer is
not enumerable:

- **Anomaly detection.** "Is this specimen normal?" — labels are
  structurally insufficient; the FM's representation must encode
  what "normal" means for the dataset. Compare typed-output
  baselines against representation-aware baselines on novel-motif
  detection.
- **Interpolation.** "Generate a specimen halfway between
  solid-ring and liquid-tri-disk." — output is a specimen, not a
  label. Connector tokens, joint training, and FM-LLM
  compositionality can show their value.
- **Active experimental design.** "Which measurement would best
  disambiguate these two hypotheses?" — output is a query,
  evaluated by information gain. Rich representations enable
  principled active learning.
- **Hypothesis generation in scientific dialogue.** Multi-turn
  refinement with a human-in-the-loop. Open-ended; rich
  representation drives hypothesis space.

**Why impactful:** the field's open architectural questions
(connector tokens, joint training, differentiable composition,
SAE steering) plausibly resolve differently in this regime. Our
negatives don't carry over. New testbeds are where the field
moves.

**Cost:** rebuild testbed infrastructure (~3-6 months of work);
new evaluation metrics; domain expertise to validate
"interesting" outputs. Highest payoff: a paper-worthy contribution
on the dependence of LLM-FM architectural choices on task type.

## My recommendation

If you have ~1 person-month: do **A** (CoT scale-up) and **D**
(clean `full_probes`) for tighter numbers, plus **C**
(coefficient sweep) to firm up the Phase 15 negative.

If you have ~1 person-quarter: do **B** (SAE at scale) — it's
the cheapest unlock that could materially flip the Phase 15
finding.

If you have ~1 person-year: do **Bet 3** (reframe the testbed).
The narrow-classification regime is exhausted; the field's
interesting questions live in the open-ended regimes our testbed
can't touch.

If you have ~2-3 person-years: do **Bet 2** (causal abstraction).
It's the deepest open problem connected to this work, and the
least crowded research direction in the field.

## What I would explicitly *not* do

- **Run more representation-injection variants** (different probe
  banks, different SAE architectures, alternative connector
  designs) on the same testbed. Six negatives is enough; the
  marginal information from a seventh is small.
- **Try to push `full = 0.695` higher with verifier tuning.** The
  ceiling is task-bound; squeezing 0.71 out of 0.695 has low
  scientific value compared to either deeper interpretability or
  a different testbed.
- **Spend more compute on a 200-row SAE.** Either scale to
  Templeton-region or accept Phase 15 as a small-data result.

## Connecting to the broader research landscape

The directions above map to active 2025-2026 research areas:

- **Bet 1** ↔ Anthropic's "Tracing the thoughts of an LLM"
  (Lindsey et al. 2025) and OpenAI's circuits work.
- **Bet 2** ↔ Geiger et al.'s causal abstraction work, Wu et al.'s
  interchange interventions.
- **Bet 3** ↔ Active research on scientific-discovery agents
  (FunSearch, MathTotem, AlphaProof) and on dialogue benchmarks
  (MMLU-Pro, ChatArena, scientific QA datasets).

A new agent picking this up has the option to ride one of these
waves rather than work in isolation.
