# 04 — Experiments and Results

Every architectural variant tested, with hypothesis, method, and
result. Held-out range `[40000, 40200)` throughout. All numbers
are `goal_accuracy` unless noted.

## Reference axis: the Phase 8a baselines

Three baselines define the floor and ceiling of the rest of the
project:

| Baseline | What it is | Goal accuracy |
|---|---|---|
| `naked` | One-shot LLM commit. No FM, no verifier, no observations. | 0.000 |
| `no_verifier` | OHVD loop with FM tools, but verifier always returns PASS. | 0.540 |
| `full` | Full Pipeline A: FM tools + multi-source verifier (V4). | **0.695** |

The architectural lift decomposes:

- `naked → no_verifier`: +0.540 (FM tools and typed-evidence reasoning).
- `no_verifier → full`: +0.155 (multi-source verifier).

Verdict-stratified at `full`:

| | full |
|---|---|
| commit_rate | 1.000 |
| hallucination_rate (wrong-PASS / total-PASS) | 0.255 |
| calibrated_abstention (wrong-CAVEAT / total-wrong) | 0.590 |
| verdict P/C/N | 98 / 102 / 0 |

## Phase 9 — connector tokens (Layer C)

**Hypothesis:** the LLM can reason better if it receives FM2's
hidden state as soft tokens (continuous embeddings projected into
LLM token space) than as typed JSON.

**Method:** trained a connector network that maps FM2 CLS
embedding to a sequence of soft tokens, then prepended those
tokens to the LLM's input. Trained the connector with an
alignment objective on templated text targets.

**Result:** below `full`. Closed as a documented negative.

**Mechanism:** the LLM's training distribution doesn't include
soft-token connectors with this specific learned projection. The
alignment objective rewarded matching templated text, not solving
the task. Net negative.

**Reference doc:** `docs/progress/09-connector.md`,
`docs/audits/09-connector-audit.md`.

## Phase 10 — SSL backbone (Layer D)

**Hypothesis:** a richer FM2 backbone — pretrained with masked-RDF
reconstruction instead of supervised energy regression — would
provide a more informative representation for downstream probes
and connectors.

**Method:** masked 30% of `g(r)` bins, trained FM2 to reconstruct
them (BERT-style). Compared the resulting CLS embedding's probing
performance against the supervised one.

**Result:** the SSL representation produced *poorer* probing
performance than supervised. Closed as a documented negative.

**Mechanism:** masked-RDF reconstruction rewards capturing
correlations between bins; energy regression rewards capturing
features that drive total energy. The latter happens to also be
more useful for downstream probes (atom count, motif). The
representation that's "richer" by reconstruction loss is
"poorer" by downstream task transfer.

**Reference doc:** `docs/progress/10-ssl-fm2.md`,
`docs/audits/10-ssl-audit.md`.

## Phase 11 — CoT-SFT bootstrap

**Hypothesis:** if we train the LLM (LoRA) on synthetic
chain-of-thought traces that explicitly read probe outputs and
reason over them, the trained adapter will do tool-grounded
reasoning at inference.

**Method:** Stage 0 trained five probes on FM2 CLS (`n_atoms`,
`motif`, `phase`, `coordination`, `peak_position`). Stage 1
generated synthetic CoT records: deterministic four-step
reasoning chains that mention probe values, do a physical
cross-check, and commit ground truth. Stage 2 SFT'd Qwen on
~10K such records. Phase 11.B evaluated on held-out.

**Result:** `cot_sft = 0.467`. Below `no_verifier = 0.540`.
Below `full = 0.695` by 23 points.

**Mechanism:** the trained adapter learned to reference probe
outputs in its CoT, but probe outputs are a lossy summary of the
bridged FM tool messages the standard pipeline gives the LLM. The
adapter underperforms the un-adapted LLM with full bridged
evidence.

**Reference doc:** `docs/progress/11-cot-sft.md`,
`docs/audits/11-cot-sft-audit.md`.

**Caveat:** Stages 3 (rejection sampling) and 4 (GRPO with
verifier reward) were intentionally not built. Their effect is
unmeasured.

## Phase 12 — probe injection (Pipeline A + probes in prompt)

**Hypothesis:** if probes summarize FM2 succinctly, injecting
them into Pipeline A's user message (alongside the bridged tool
messages) gives the LLM richer evidence and should lift goal
accuracy.

**Method:** computed the five probe outputs per specimen and
prepended them to the OHVD loop's enriched query string.
Otherwise vanilla Pipeline A.

**Result:** `full_probes = 0.562`. Below `full = 0.695`.

(Caveat: this number was computed over a JSONL that the resume
logic accidentally appended a duplicate pass into, yielding 400
lines for 200 specimens. The qualitative finding holds — adding
probes hurts vs vanilla `full` — but the specific 0.562 has
duplication contamination. Re-run cleanly before reporting it.)

**Mechanism:** the LLM receives both the typed FM tool messages
*and* the probe summaries. The probe summaries are a lossy view
of what the tool messages already cover. Two competing evidence
streams confuse rather than enrich.

**Reference doc:** `docs/progress/12-full-probes.md`.

## Phase 13 — SAE feature labels in prompt

**Hypothesis:** auto-discovered SAE features over FM2 CLS may
expose information the hand-picked probes miss. Surfacing the
top-k active labelled features per specimen as text in the LLM's
prompt should help.

**Method:** Stage 0 trained a Top-K SAE (in_dim=320, hidden_dim=1024,
k=32) on FM2 CLS over 20K training specimens. Stage 1 labelled
each feature by correlation with motif / atom count / temperature
/ phase. Stage 2 ran Pipeline A with the top-8 active labelled
features for each specimen injected into the user message.

**Result:** `full_sae = 0.585`. Below `full = 0.695` by 11 points.

**Mechanism:** same as Phase 12 — the LLM treats the SAE
labels as facts, anchors on them, and reasons less from the
bridged FM messages. Hallucination_rate goes up (0.255 → 0.286)
specifically on the CAVEAT bucket where the LLM has to reason
across uncertainty.

**Reference doc:** `docs/progress/13-sae.md`.

## Phase 14 — causally-filtered SAE labels

**Hypothesis:** Phase 13's SAE labels are correlation-based —
they describe what activates each feature, not what it does. A
causal filter (knock-out interventions on the SAE latent → measure
ΔE in FM2's energy head; keep only features with non-trivial
causal effect) would eliminate decorative features and surface
only the ones FM2 actually uses, plausibly closing the gap to
`full`.

**Method:** Phase 14 audit script forwards FM2 over a 2K-specimen
audit set, runs knock-out / knock-in interventions on every SAE
feature, computes signed effect = `intervened - recon` energy
difference normalized by inter-specimen std. Filter at
`|effect_norm| >= 0.10` plus `activation_rate >= 0.005`. Then
re-run Phase 13's prompt-injection pipeline restricted to passing
features.

**Result:** `full_sae_causal = 0.570`. Slightly *worse* than
`full_sae = 0.585`. Both below `full = 0.695`.

**Mechanism:** the causal filter dropped some features the LLM
was extracting useful signal from. Causal validity by
intervention on FM2's energy head doesn't equal informativeness
for the LLM's classification reasoning. The two objectives
diverge.

**Reference doc:** `docs/progress/14-sae-causal.md`.

## Phase 15 — SAE on the LLM (Golden Gate Claude analog)

The longest single phase. Four sub-stages.

### Stage A: harvest Qwen residual activations

Replay 200 trajectories from a prior `full` run, reconstruct the
minimal chat `[system, user, assistant=final_claim]` for each,
forward through Qwen with a hook on `model.layers.14`, capture
the last-token residual. Output: `(200, 3584)` activation matrix
+ per-row metadata (verdict, is_correct, motif, phase, N, T).

### Stage B: Top-K SAE on the harvested activations

`hidden_dim = 16384`, `k = 64`, 30 epochs. Final loss `0.097` at
50 steps (200 rows / 128 batch / 30 epochs ≈ 50 batches).

### Stage C: label features by verdict / correctness / motif / phase / N / T

Output: 476 features locked on at least one axis, 8 unlabelled,
15900 rare. Steering candidate counts: 7 wrong-PASS features
(features whose top activators are all PASS commits AND all
wrong commits), 79 wrong-any features, 106 caveat features.

The seven wrong-PASS candidates *all* lock onto
`motif=triangular_disk` (and most onto `phase=solid-like`):

| fid | verdict purity | correct purity | n_top | label |
|---|---|---|---|---|
| 4741 | 1.00 | 1.00 | 11 | tri_disk + correct=False |
| 6844 | 1.00 | 1.00 | 11 | tri_disk + solid + correct=False |
| 7917 | 1.00 | 1.00 | 10 | tri_disk + solid + correct=False |
| 11357 | 0.85 | 0.77 | 13 | tri_disk + correct=False |
| 3631 | 0.80 | 0.73 | 15 | tri_disk + correct=False |
| 7592 | 0.73 | 0.73 | 15 | tri_disk + correct=False |
| 11764 | 0.79 | 0.71 | 14 | tri_disk + correct=False |

### Stage D: steered Pipeline A

Ran with `fid=6844, coefficient=-2.0` (ablate the cleanest-locked
candidate).

**Result:** `full_steered_6844_n200 = 0.660`. Below
`full = 0.695` by 3.5 points. **Steering this feature hurt.**

The verdict mix matched `full` exactly (98 PASS / 102 CAVEAT) so
the verifier was identical. But:

| | full | full_steered_6844_n200 | Δ |
|---|---|---|---|
| hallucination_rate | 0.255 | 0.316 | +0.061 |
| calibrated_abstention | 0.590 | 0.544 | -0.046 |
| wrong-PASS count | 25 | 31 | +6 |

**Mechanism (most likely):** `fid=6844` is a useful
*tri-disk-solid detector*, not a wrong-commit cause. The SAE label
"top activators are wrong tri-disk-solid" is correlation-based
and doesn't say "this feature causes wrong commits." Ablating a
useful detector flipped some of the 56 right tri-disk-solid
commits to wrong, while the 8 already-wrong tri-disk-solid
specimens didn't get fixed because the feature isn't what was
making them wrong.

This is the **interpretive lesson of Phase 15**: correlation-based
SAE labels describe what activates a feature, not what its causal
role is. Steering by label produces the wrong intervention
direction.

A diagnostic of the source `full` run revealed the labels were
also misleading geographically: the wrong-PASS commits are
concentrated on `(ring, *)` (8/8 wrong = 100%) and
`(triangular_disk, liquid-like)` (18/26 = 69% wrong), while
`(triangular_disk, solid-like)` is the *safest* group (12% wrong)
— but that's where every clean SAE feature locked, because
tri-disk-solid dominates the dataset volume (64 of 98 PASS).

**Reference doc:** `docs/progress/15-qwen-sae.md`.

**Caveats on Phase 15:** SAE was trained on 200 rows. Templeton
et al. used ~1B activations. Our negative is "this specific
small-data SAE doesn't help via this specific feature" — not
"SAE steering doesn't work on LLM-orchestrated tool use."

## Phase 16 — SAE-augmented CoT-SFT, no verifier (the positive)

**Hypothesis:** an LLM trained via SFT on synthetic CoT records
that include both probe outputs *and* SAE-derived feature labels
can outperform the FM's own downstream head and the probes-only
CoT-SFT baseline on a single-shot, no-verifier classification task.

**Method:** extended `synthetic_cot.py` to render Step 1
(probes) and Step 1b (top-K labelled SAE features) in the chain.
Built a 10K-record dataset, trained a fresh LoRA adapter via SFT
(4-GPU DDP, 3 epochs, effective batch 32, MAX_SEQ=1536), then
evaluated single-shot with bf16 inference and MAX_NEW_TOKENS=768.
Also added a reference baseline `probe_head` that maps probe
outputs directly to a `PhysicalStateClaim` with no LLM.

**Result (positive):**

| | goal_accuracy | Δ vs cot_sft_sae |
|---|---|---|
| `probe_head` | 0.110 | +54.0 (LLM beats FM head by huge margin) |
| `cot_sft` | 0.467 | +18.3 (SAE features add real training signal beyond probes) |
| **`cot_sft_sae`** | **0.650** | reference |
| `full` | 0.695 | -4.5 (single-shot LLM with no verifier nearly matches full Pipeline A) |

**Sanity:** 200/200 committed, 198/200 unique final_claims (no
template memorization). Per-(motif, phase) decomposition shows
the failure pattern *shifted*: ring buckets that were 0% under
`full` are now 33-43%; tri-disk-solid (the easy group) drops
from 88% → 76%. The wrong-claim sets of `full` and
`cot_sft_sae` are qualitatively different.

**Mechanism:** training-time SAE label injection via supervised
CoT works where inference-time injection (Phase 13/14) failed.
The LLM, when *taught* to read SAE labels in the synthetic CoT
chain, extracts useful information they encode about ring
discrimination. When asked to read them in-context (Phase 13),
it anchored on them and reasoned worse.

**Reference doc:** `docs/progress/16-cot-sft-with-sae.md`.

## Side-by-side comparison (current state)

```
HEADLINE:
  naked              = 0.000
  naked_vision       = 0.000
  probe_head         = 0.110  (FM head, no LLM)
  cot_sft            = 0.467
  no_verifier        = 0.540
  full_probes        = 0.562  (n=400, duplicates)
  full_sae_causal    = 0.570
  full_sae           = 0.585
  cot_sft_sae        = 0.650  *** Phase 16 positive
  full_steered_6844  = 0.660
  full               = 0.695  ← architectural ceiling (with verifier)
```

Verdict-stratified breakdown for the verifier-using rows:

| | full | full_probes | full_sae | full_sae_causal | full_steered_6844_n200 |
|---|---|---|---|---|---|
| commit_rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hallucination_rate | 0.255 | 0.310 | 0.286 | 0.286 | 0.316 |
| calibrated_abstention | 0.590 | 0.669 | 0.663 | 0.674 | 0.544 |
| verdict P/C/N | 98/102/0 | 187/213/0 (×2 dup) | 98/102/0 | 98/102/0 | 98/102/0 |

For the no-verifier rows (`cot_sft`, `cot_sft_sae`,
`probe_head`, `naked`, `naked_vision`), `verdict P/C/N = 0/0/N`
because no verifier was run; `commit_rate` indicates parse
success and `goal_accuracy` is computed against ground truth
directly.

## Eight world-model tests (auxiliary)

`scripts/run_evaluation.py` produces eight test scores per
baseline. The headline is `goal_accuracy`. The other seven
(`trajectory_compression`, `trajectory_distinction`,
`step_recoverability`, `prediction_compression`,
`prediction_distinction`, `goal_competence`,
`federated_factorability`, `calibrated_uncertainty`) are
diagnostic and supplementary; they do not change the headline
ordering. Most baselines pass them; `naked` and `naked_vision`
fail several because they emit a degenerate identical commit on
every specimen.

## What was *not* tried

These were either explicitly out of scope or deferred:

- **Stage 3 (rejection sampling) and Stage 4 (GRPO with verifier
  reward) for Phase 11.** Could close some of the 23-point gap.
- **Frontier LLMs** (Claude 4.x Opus, GPT-4-class, o1). Phase 13
  / 14 / 15 may behave very differently with stronger reasoning
  models.
- **SAE at scale** (10K-1B+ rows). Templeton-scale would change
  Phase 15 conclusions.
- **Multi-token activation harvest** in Phase 15 Stage A. Capturing
  all assistant tokens would 50-200x our SAE training data
  cheaply.
- **Coefficient sweep** in Phase 15 Stage D. Only `coef=-2.0` on
  `fid=6844` was run. The plan included `+2.0` on the same fid
  (testing the "useful detector" hypothesis) and `-2.0` on
  `fid=11357` and `fid=16320` (broader-target features).
- **Open-ended discovery tasks**, where the testbed's discretized
  output space doesn't saturate the labels. See
  `07-open-questions-and-bets.md`.

## Status of artefacts on disk

These exist on the remote and are referenced by all the numbers
above:

```
checkpoints/
  fm1_*/                     # FM1 trained checkpoint
  fm2_rdf/                   # FM2 trained checkpoint
  fm3_*/                     # FM3 trained checkpoint
  probes/                    # Phase 11 probe bank
  cot-sft/                   # Phase 11 LoRA adapter
  cot-sft-sae/               # Phase 16 LoRA adapter (SAE-augmented)
  sae/                       # Phase 13 FM2 SAEs (multiple runs)
  qwen_sae/                  # Phase 15 Qwen SAE

runs/
  baselines/full/            # Phase 8a training-set runs
  holdout/                   # all held-out baselines
    naked/
    naked_vision/
    no_verifier/
    cot_sft/
    full/
    full_probes/             # has dup contamination, see above
    full_sae/
    full_sae_causal/
    full_steered_6844_n200/
    cot_sft_sae/             # Phase 16 cot_sft_sae trajectories
    probe_head/              # Phase 16 FM-head reference
  cot_datasets/              # Phase 11 synthetic CoT records
  cot_datasets_sae/          # Phase 16 SAE-augmented CoT records
  sae_labels/                # Phase 13 feature labels
  sae_causal/                # Phase 14 causal effects + filter
  qwen_activations/          # Phase 15 Stage A
  qwen_sae_labels/           # Phase 15 Stage C
  comparisons/               # side-by-side reports
  eval/                      # individual report.yamls

data/
  synthetic_lj_v1/specimens.h5
  synthetic_lj_v1/splits.yaml
  literature/clusters.json
```
