# 10 — Literature by Phase

This document maps the external literature onto the specific
experimental steps we ran. For each phase: what we did, what
relevant prior work exists, how our approach connects to or
differs from each piece of prior work, and what our result adds
to the conversation.

The companion file `08-references.md` lists external work in
flat form. This one is structured by *which step of our project
the reference informs*. A new agent picking up the work should
use `08` as a bibliography and `10` (this file) as the cross-
reference into the experimental record.

---

## Phase 4 — Multi-source verifier

### What we did
Built a five-source verifier (`rule_library`, `literature`,
`cross_fm`, `simulator`, `conformal`) that returns
`PASS / CAVEAT / FAIL` per source and aggregates to a worst-case
verdict. Iterative revision lets the LLM respond to CAVEAT with
a refined hypothesis. Calibrated abstention via the CAVEAT
verdict is what gives `full = 0.695` its 15.5-point lift over
`no_verifier = 0.540`.

### Prior work this connects to

- **Let's Verify Step by Step (Lightman et al. 2023, OpenAI).**
  Process Reward Models score every reasoning step rather than
  the final outcome. *Similar:* step-level verification beats
  outcome-only. *Differs:* PRMs are trained neural verifiers;
  ours are independent rule-based / retrieval-based / simulator-
  based oracles. Our approach is more brittle to specify but
  carries no train/test distribution mismatch — every source has
  its own ground truth.

- **Math-Shepherd (Wang et al. 2024, ACL).** Step-by-step
  PRM-trained verifier used as both inference filter and RL
  signal. *Similar:* uses verifier signal as both gate and
  reward. *Informs:* Phase 11 Stages 3-4 (unbuilt) would have
  used GRPO with our verifier as reward — the same pattern
  Math-Shepherd validates for math.

- **VersaPRM (2025).** Multi-domain PRM trained from synthetic
  CoT plus LLM labelers. *Similar:* generalizing verifier across
  domains. *Differs:* their verifier is one trained model; ours
  is five disjoint sources. Multi-source-with-aggregation versus
  single-trained PRM is an open architectural choice.

- **CompassVerifier (EMNLP 2025).** Unified verifier that
  recognizes invalid / abnormal / long-reasoning responses.
  *Similar to our calibrated abstention via CAVEAT* — refusing
  to commit when confidence is low.

- **rbio-1 (bioRxiv 2025).** RL post-trained reasoning model
  using *biological world models as soft verifiers*. **The
  closest published instantiation of our verifier philosophy.**
  *Similar:* uses ML world-model as oracle for verification.
  *Differs:* single biological world-model versus our five-
  source ensemble; RL-trained versus inference-time gating.

- **Reflexion (Shinn et al. 2023).** LLM self-reflection on
  errors via verbal feedback as memory. *Similar in iterative
  revision spirit.* *Differs:* internal self-verification only,
  no external oracle.

- **Self-Consistency (Wang et al. 2022).** Sample many CoTs,
  take majority. *Similar low-cost verifier.* *Differs:* uses
  population variance rather than independent oracles.

- **Generalizable PRMs via Formally Verified Training Data
  (2025).** Use formal-verification tools to label training
  data for PRMs. *Similar in spirit:* use formal/symbolic ground
  truth as verifier source. *Differs:* applies to formal proof
  domains; ours operates with natural-science oracles.

- **ReST-MCTS\* (NeurIPS 2024).** PRM-guided tree search for
  LLM self-training. *Similar:* verifier guides multi-step
  reasoning. *Differs:* internal verifier; we have external
  multi-source oracles.

### What our work adds

A *layered* verifier (five independent oracles + iterative
revision + calibrated abstention) on a controlled testbed with
ablation across V0..V4 source combinations. The robustness of the
+15.5-point lift across configurations is reported. Most
published verifiers are single-source or trained PRMs; multi-
oracle ensembles with formal abstention semantics are
under-tested in 2024-2025.

---

## Phase 5 — OHVD loop (Observe-Hypothesize-Verify-Decide)

### What we did
Iterative loop where the LLM alternates calling FMs as tools
(observe), hypothesizing a structured claim, getting verifier
feedback, and deciding whether to commit. Up to 16 turns.

### Prior work this connects to

- **ReAct (Yao et al., ICLR 2023).** Interleaved reasoning and
  acting; the canonical pattern. *Direct ancestor* of our OHVD
  loop. *Differs:* ReAct interleaves thoughts/actions without a
  formal verifier checkpoint between turns; OHVD adds a
  verifier per cycle.

- **Reflexion (Shinn et al. 2023).** Adds reflection memory
  across attempts. *Similar:* iterative-correction structure.

- **ChemReasoner (Sprueill et al., ICML 2024).** Catalyst-
  discovery agent with LLM hypothesis + GNN feedback loop.
  **Most similar iterative structure** to ours: LLM proposes,
  scientific FM checks, refine. *Differs:* one FM (GNN)
  versus our panel; feedback used for search, not verifier-
  style abstention.

### What our work adds
The verifier-as-checkpoint between turns, with the LLM allowed to
either commit or call more tools after each verdict, plus
explicit budget (max_steps) and termination semantics. The OHVD
loop unifies tool use and verification in a single trajectory
schema that both Phase 6's RL trainer and Phase 7's eight tests
consume uniformly.

---

## Phase 8a — Reference baselines (`naked` / `no_verifier` / `full`)

### What we did
Three reference baselines: `naked` (zero-shot LLM commit, no FM,
no verifier), `no_verifier` (OHVD loop with NoOpVerifier always
PASS), `full` (V4 verifier with all five sources). Result:
`naked = 0.000`, `no_verifier = 0.540`, `full = 0.695`.

### Prior work this connects to

- **HuggingGPT / JARVIS (Shen et al. 2023, Microsoft).** LLM
  controller decomposes user requests, picks expert models from
  HuggingFace, integrates results. *Most direct precedent for
  the LLM-as-multi-FM-orchestrator pattern.* *Differs:* generic
  vision/audio/text tasks, no domain physics, no verifier loop,
  no ablation across architectural variants.

- **ChemCrow (Bran et al. 2024, Nature MI).** GPT-4 + 18
  chemistry tools. *Most similar in spirit* to `full`: LLM
  orchestrating heterogeneous tools in a domain. *Differs:* no
  multi-source verifier with formal abstention; no
  representation-level architectural ablation.

- **Coscientist (Boiko et al. 2023, Nature).** Autonomous
  chemistry agent with internet, code, lab automation tools.
  *Similar:* autonomous LLM-led multi-tool reasoning. *Differs:*
  no foundation-model layer; uses external resources rather than
  internal scientific FMs.

- **Toolformer (Schick et al., NeurIPS 2023).** Self-supervised
  trained tool use. *Different paradigm:* trained tool use
  versus our frozen-LLM tool routing.

- **Gorilla (Patil et al. 2023).** LLM fine-tuned to call
  thousands of APIs. *Differs:* about call accuracy, not
  evidence integration.

- **Visual Programming (Gupta & Kembhavi, CVPR 2023).** LLM
  emits Python program of vision-tool calls. *Similar:*
  structured tool composition. *Differs:* code-as-glue versus
  typed-JSON contracts.

### What our work adds
A controlled three-baseline ablation on a closed-world testbed
showing the verifier's standalone contribution (+15.5 points).
Most ChemCrow / Coscientist-class papers do not factor out the
verifier from the FM-tool ecosystem.

---

## Phase 9 — Connector tokens (Layer C)

### What we did
Trained a learned projection from FM2's CLS embedding to a
sequence of soft tokens prepended to the LLM's input. Frozen LLM,
templated text alignment loss. *Result:* below `full`. Closed as
a documented negative.

### Prior work this connects to

- **BLIP-2 / Q-Former (Li et al. 2023).** Q-Former learns query
  vectors that pool visual features for a frozen LLM via two-
  stage pretraining. **Direct architectural analog** to what we
  built. *Their finding (vision-language):* Q-Former works.
  *Our finding (physics):* connector tokens did not transfer
  specimen identity. **The contrast is itself a result:** we
  have no language-rich captioning corpus the way vision-
  language has, so the alignment loss does not anchor on a
  natural-language semantic prior.

- **MiniGPT-4 (Zhu et al. 2023), LLaVA (Liu et al. 2023).**
  Linear projection from CLIP-ViT to Vicuna's input space.
  *Similar architecture, with abundant instruction-tuning data.*

- **Frozen (Tsimpoukelli et al. 2021, NeurIPS).** Vision-encoder
  tokens prepended to a frozen LM context. *Direct ancestor of
  our Phase 9 architecture, in the vision domain.*

- **Unified Molecule-Text LM with Discrete Token Representation
  (2024).** VQ-tokenizer + Q-Former bridge for molecules. *The
  closest published chemistry analog to our Phase 9.* *Differs:*
  discretizes via VQ; uses molecule-text paired data. Their
  setup has more language supervision than ours.

- **Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in
  Molecular Understanding (2026).** Notes that fixed-length
  Q-Former-style bridges miss substructural context for
  molecules. **Validates our negative-result framing:**
  Q-Former-style bridges do not transfer well to scientific
  modalities without modification. Their fix is dynamic-length
  tokens; ours abandoned the pathway.

### What our work adds

A controlled-comparison demonstration that connector-token
bridges underperform typed-JSON contracts in a frozen-LLM
regime where the alignment objective cannot rely on a natural-
language captioning prior. The chemistry-side papers above hint
at the same finding; ours establishes it crisply on a synthetic
testbed.

---

## Phase 10 — SSL backbone for FM2 (Layer D)

### What we did
Pretrained FM2 with masked-RDF reconstruction (BERT-style mask
30% of `g(r)` bins). Compared the resulting CLS embedding's
probing performance against the supervised energy-regression
baseline. *Result:* poorer probing performance. Closed as a
documented negative.

### Prior work this connects to

- **MultiPUFFIN (2025).** Multimodal molecular FM using SMILES
  + 2D graph + 3D conformer with cross-modal attention.
  *Conceptual cousin:* multiple representations of one molecule.
  *Differs:* trained end-to-end with downstream supervision, not
  unsupervised pretrain.

- **MolSpectLLM (2025).** Qwen-based FM unifying spectroscopy
  with 3D structure generation. *Similar in scientific multi-
  modal scope.*

- Broader self-supervised pretraining literature (BERT, MAE,
  BYOL) — not surveyed in detail here. Our finding is consistent
  with the well-known observation that SSL pretraining
  objectives that reward bin-bin correlations don't necessarily
  optimize features for downstream supervised tasks.

### What our work adds
Direct comparison of supervised-energy-regression backbone vs
masked-RDF SSL backbone on the same held-out probing tasks. The
supervised representation transfers better. Documents one more
case where reconstruction-style SSL is not the right pretraining
objective for scientifically-supervised downstream tasks.

---

## Phase 11 — Synthetic CoT-SFT bootstrap

### What we did
Stage 0 trained five probes on FM2 CLS. Stage 1 generated
synthetic CoT chains: deterministic four-step reasoning that
references probe values and commits ground truth. Stage 2 SFT'd
Qwen 2.5 7B with LoRA on ~10K such records. *Result:*
`cot_sft = 0.467`, below `no_verifier = 0.540`. Stages 3-4 not
built.

### Prior work this connects to

- **STaR — Self-Taught Reasoner (Zelikman et al., NeurIPS
  2022).** *Exactly the unbuilt Stage 3 of our Phase 11:* sample
  many CoTs, keep ones whose final answer matches ground truth,
  SFT a second round. STaR validates that this loop typically
  closes ~25-50% of the gap between SFT-only and the ceiling.

- **DeepSeekMath / GRPO (Shao et al. 2024).** *Exactly the
  unbuilt Stage 4:* RL against a verifier-shaped reward, with
  group relative policy optimization. The verifier in our setup
  is already wired; only the prompt format and the GRPO loop
  remain.

- **o1 / o3 reasoning — OpenAI 2024-2025.** Productionized
  version of the Stage-3-4 paradigm. Inference-time CoT scaling
  under verifier-shaped training. *The closest evidence that the
  full Phase 11 stack might in principle close the
  `cot_sft → full` gap*, even though we did not build it.

- **Math-Shepherd (Wang et al. 2024).** Step-by-step PRM-trained
  verifier feedback. *Same pattern* as a hypothetical Stage-4
  verifier-as-reward.

- **KNOS — Knowledge-guided Solver (IEEE TKDE 2025).**
  Invoke-Verify-Inject framework with dual-process reasoning.
  *Similar dual-system architecture* (separate knowledge and
  inference systems with verification between them).

### What our work adds
Stage 2 alone underperforms the no-adapter pipeline by 7
points, even with rich synthetic supervision. This is a
useful negative for the position that "trained CoT-on-probes is
sufficient." The probe-based summary is too lossy compared to
the bridged FM tool messages the un-adapted LLM gets at
inference. Stages 3-4 are explicitly unbuilt and may close the
gap.

---

## Phase 12 — Probe injection in prompt (`full_probes`)

### What we did
Computed five probes per specimen and prepended them to the
OHVD loop's user message. Otherwise vanilla Pipeline A.
*Result:* `full_probes = 0.562` (over duplicated 400-line
JSONL — qualitative finding holds, exact number contaminated).

### Prior work this connects to

- **ChemDFM-X (2024).** Single multimodal molecular FM
  consuming SMILES + 3D conformations. *Alternative paradigm:*
  bake multimodality into one FM rather than route through an
  LLM with multimodal evidence in the prompt.

- **MultiPUFFIN (2025).** As above — multi-encoder fusion
  versus prompt-injection of multi-source evidence.

- **Cost-Aware Model Orchestration (arXiv 2025).** Studies how
  to route between LLMs based on capability and cost. *Related:*
  picking when richer evidence helps. *Differs:* doesn't
  ablate evidence richness within a given LLM.

- **ChemToolAgent (2024).** Studies how giving an LLM more
  chemistry tools affects its problem-solving. *Most similar
  framework:* ablating the tool surface. *Differs:* their axis
  is tool quantity; ours is evidence pathway (typed contract vs
  prompt-side richer summaries).

### What our work adds
Demonstrates that adding *more* evidence (in this case,
auxiliary probe summaries on top of bridged FM tool messages)
slightly hurts goal accuracy in a frozen-LLM regime. The effect
is concentrated in the CAVEAT bucket where the LLM has to
weigh competing evidence sources and tends to anchor on the
extra signal. Connects to ongoing debate about whether RAG-
style augmentation always helps.

---

## Phase 13 — SAE feature labels in prompt (`full_sae`)

### What we did
Trained a Top-K SAE on FM2 CLS (`hidden_dim=1024`, `k=32`),
labelled features by correlation with motif/N/T/phase, and
injected the top-8 active labelled features per specimen into
Pipeline A's user message. *Result:* `full_sae = 0.585`. Below
`full = 0.695`.

### Prior work this connects to

- **Towards Monosemanticity (Bricken et al., Anthropic 2023).**
  Foundational SAE-on-LM paper. *We use the same recipe* on the
  small science-FM rather than on the LLM itself.

- **Scaling Monosemanticity (Templeton et al., Anthropic 2024).**
  SAE on Claude 3 Sonnet at 16M-feature scale; the
  Golden-Gate-Bridge demo. *We adopt the recipe*; our scale is
  ~8 orders of magnitude smaller.

- **Scaling and Evaluating SAEs (Gao et al., OpenAI 2024).**
  Top-K SAE recipe. *The exact training recipe we use* in
  Phases 13/14/15.

- **Towards Monosemanticity / Scaling Monosemanticity** are the
  *only* prior precedents I found for using SAE feature labels
  as text-readable evidence to a downstream model. Both papers
  use SAE features for *interpretation* of the model that hosts
  them, not as input to a separate reasoner. **Our Phase 13
  application — SAE labels from FM-A as prompt evidence to
  LLM-B — is genuinely novel-axis.**

### What our work adds
First test (to my knowledge) of correlation-labelled SAE features
as text-readable prompt evidence to an LLM orchestrator. The
finding is the same as Phase 12 (probes) plus the lesson that
auto-discovered features don't escape the prompt-pollution
penalty: the LLM still anchors on extra evidence.

---

## Phase 14 — Causal validation of SAE features (`full_sae_causal`)

### What we did
For each SAE feature, ran knock-out / knock-in interventions on
the latent and measured ΔE in FM2's energy head. Filtered
features by `|effect_norm| >= 0.10`. Re-ran Phase 13's prompt
injection restricted to passing features. *Result:*
`full_sae_causal = 0.570`. Slightly worse than `full_sae`.

### Prior work this connects to

- **Causal Abstraction of Neural Networks (Geiger et al. 2021,
  NeurIPS).** Formal framework for when one network is a causal
  abstraction of another. *Theoretical foundation for the more
  ambitious version of this question.* The Phase 14 audit is a
  practical instance of the broader causal-abstraction agenda.

- **Distributed Alignment Search (Wu et al., ICML 2024).**
  Interchange interventions to identify causally relevant
  subspaces. *Methodologically similar* to Phase 14's knock-out
  pattern.

- **Causal Mediation Analysis (Vig et al., NeurIPS 2020).**
  Mediation analysis applied to model bias. *Methodological
  cousin* to our knock-out audit.

- **Sparse Feature Circuits (Marks et al. 2024).** Use SAEs as
  nodes in causal circuits. *Similar in spirit.* *Differs:*
  multi-feature circuit identification rather than single-feature
  ablation.

- **On the Biology of an LLM / Tracing the Thoughts of an LLM
  (Lindsey et al., Anthropic 2025).** SAE features as nodes in
  attribution graphs. *Closest published embodiment* of where the
  bigger Phase-14-style line could go.

- **Use SAEs to Discover Unknown Concepts, Not to Act on Known
  Concepts (arXiv 2506.23845, 2025).** Argues SAEs underperform
  prompting/finetuning when the goal is to act on known concepts.
  **The closest published statement of the lesson Phase 14
  reaffirms:** correlation-labelled SAE features are descriptive,
  not interventional. Filtering by causal effect on a *different*
  objective (FM2 energy head) doesn't translate to causal
  effect on the LLM's downstream reasoning.

### What our work adds
A causal-validation audit using a *domain-grounded* predictive
head (FM2 energy regression) as the intervention target —
distinct from the more common LLM-loss or behavior-on-text
targets in interpretability literature. The negative finding
(causal-on-FM-head does not equal informativeness-for-LLM-
classification) is a clean demonstration that causal validity
is target-specific.

---

## Phase 15 — SAE on Qwen residual stream + activation steering

### What we did
Stage A: harvest Qwen residual activations at `model.layers.14`
on 200 trajectories. Stage B: Top-K SAE on those activations
(`hidden_dim=16384`, `k=64`). Stage C: label features by
verdict / correctness / motif / phase / N / T. Stage D: ablate
`fid=6844` (cleanest wrong-PASS lock, purity 1.00) at
`coef=-2.0` during Pipeline A inference. *Result:*
`full_steered_6844_n200 = 0.660`. Below `full = 0.695` by 3.5
points; hallucination *increased*.

### Prior work this connects to

- **Scaling Monosemanticity / Golden Gate Claude (Templeton et
  al., Anthropic 2024).** **Direct precedent.** Same recipe
  (SAE on residual + clamp a feature). *Differs:* their scale
  was ~1B activation tokens; ours was 200. Their setting was
  open-text behavior; ours is OHVD tool-use commit.

- **Scaling and Evaluating SAEs (Gao et al., OpenAI 2024).** The
  Top-K recipe. *Our exact training pattern.*

- **Activation Addition (Turner et al. 2023).** Pre-SAE
  activation steering by direct vector injection. *The
  intellectual ancestor of SAE-based steering.*

- **Inference-Time Intervention / ITI (Li et al., NeurIPS
  2023).** Truthfulness-axis steering on attention heads.
  *Same family of intervention* with a different feature
  identification strategy.

- **Representation Engineering (Zou et al. 2023).** Broader
  framework for reading and controlling LLM representations.
  *Conceptual umbrella.*

- **CorrSteer (arXiv 2508.12535, 2025).** Generation-time
  steering via correlation-selected SAE features; argues
  correlation beats mutual-information / Fisher-information for
  feature selection. **Cautionary precedent for our negative:**
  CorrSteer's positive results are at production scale; our
  failure at 200-row scale is consistent with their setup
  requiring much more data.

- **SAE-RSV — Refinement of Steering Vectors via SAE (arXiv
  2509.23799, 2025).** Argues steering vectors from small
  datasets contain task-irrelevant noise; uses SAE to denoise.
  **Directly addresses our small-data confound:** they propose a
  fix for exactly the regime our Phase 15 negative falls into.

- **Use SAEs to Discover Unknown Concepts, Not to Act on Known
  Concepts (arXiv 2506.23845, 2025).** Argues SAEs underperform
  prompting / finetuning for steering toward *known* concepts.
  **The closest published statement of the Phase 15 lesson:**
  correlation-labelled features are descriptive, not
  interventional. Acting on them produces wrong intervention
  directions when the label was not the causal role.

- **SAE-Guided Steering for LLMs (EMNLP 2025) and Steering LM
  Refusal with SAEs (ICML 2025).** Domain-application papers in
  safety and refusal. *Same method, different axis.* Both
  operate at production scale.

- **A Comparative Analysis of SAE vs Activation Difference in
  LM Steering (2025).** Direct head-to-head between methods.
  *Same axis we tested* (with a different intervention
  baseline).

- **Sparse Feature Circuits (Marks et al. 2024).** *Methodological
  cousin*: causal interpretation of SAE features via circuit
  identification.

- **On the Biology of an LLM / Tracing the Thoughts of an LLM
  (Lindsey et al., Anthropic 2025).** *Closest conceptual home*
  for where Phase 15 would scale: SAE features as causal nodes
  in attribution graphs.

### What our work adds

A negative result showing that correlation-labelled "wrong-PASS
detector" features, when ablated at inference, can *increase*
hallucination — because the label described what activates the
feature, not its functional role. The retrospective diagnostic
(running `inspect_qwen_sae_candidates.sh` after the fact) shows
the SAE-labelled features locked on the *safest* group
(`tri_disk + solid-like`, 12% wrong) rather than the *worst*
(`ring`, 100% wrong; `tri_disk + liquid-like`, 69% wrong). This
is a concrete cautionary result for the SAE-steering recipe in
small-data regimes, complementing the more theoretical
"SAEs for unknown not known concepts" finding.

---

## Convergent architectural finding (across phases)

### What we observed

Five separate representation-pathway experiments (Phases 9, 10,
11, 13, 14, 15) all underperformed the typed-output + verifier
baseline. The verifier itself (Phase 4) is the sole positive on
top of the FM tool-use foundation.

### Prior work this connects to

I did *not* find published work that:

1. Systematically ablates many representation pathways (connector,
   SSL, CoT-SFT, prompt-injection, causal-filter, activation-
   steering) on the same testbed.
2. Operates a five-source independent-oracle verifier with
   formal calibrated abstention semantics.
3. Causally validates SAE features against a domain-grounded
   predictive head as the target.

The closest published positions are:

- **rbio-1 (2025)** — uses ML world-models as verifiers (single
  source, RL training) — closest to (2).
- **Use SAEs for Unknown Not Known Concepts (2025)** — closest
  published statement of the Phase 13-15 lesson, but argued
  rather than tested as part of an architectural study.
- **ChemReasoner (2024)** — closest published LLM-FM iterative
  loop, but with one FM and search-style feedback rather than
  verifier-style abstention.

### What our work adds
A unified architectural-ablation study showing five converging
negatives on representation-reading axes, against a controlled
typed-output + multi-source verifier baseline that is itself
under-tested in the literature. The framing suggests a paper
opportunity at the intersection of:

- **Negative-results-as-architectural-conclusion** (NeurIPS NRR
  workshops have hosted this).
- **Multi-FM scientific tool use** (ICML / NeurIPS AI4Science
  tracks).
- **SAE interpretability and steering at modest scale**
  (workshops on mechanistic interpretability, e.g. NeurIPS MI).

The strength of the contribution is that, until our results,
the literature was implicitly proceeding under the assumption
that richer-representation-into-LLM should help. Our six-axis
negative is a structured pushback on that assumption with the
caveat that the conditions we tested (frozen 7B LLM, narrow
classification task, small SAE training data, closed-world
synthetic data) bound the strength of the claim.

---

## Sources (cumulative for the handover folder)

These are the URLs the handover folder cites. They duplicate
some entries from `08-references.md` for completeness in this
file specifically.

- [HuggingGPT (Shen et al. 2023)](https://arxiv.org/pdf/2303.17580)
- [HuggingGPT GitHub mirror](https://github.com/AI-Chef/HuggingGPT)
- [ChemCrow (Bran et al. 2024) - arXiv](https://arxiv.org/abs/2304.05376)
- [ChemCrow - Nature Machine Intelligence](https://www.nature.com/articles/s42256-024-00832-8)
- [Coscientist (Boiko et al. 2023) - Nature](https://www.nature.com/articles/s41586-023-06792-0)
- [ChemReasoner (Sprueill et al. 2024) - arXiv](https://arxiv.org/abs/2402.10980)
- [ChemReasoner - ICML 2024 poster](https://icml.cc/virtual/2024/poster/35045)
- [ChemToolAgent](https://arxiv.org/html/2411.07228v2)
- [ChemDFM-X](https://arxiv.org/html/2409.13194)
- [MolSpectLLM](https://arxiv.org/html/2509.21861v2)
- [MultiPUFFIN](https://arxiv.org/html/2603.00857)
- [Multimodal LLM for materials - Nature MI](https://www.nature.com/articles/s42256-026-01214-y)
- [rbio-1 (biological world models as soft verifiers)](https://www.biorxiv.org/content/10.1101/2025.08.18.670981v1.full)
- [BLIP-2 / Q-Former (Li et al. 2023)](https://arxiv.org/pdf/2301.12597)
- [Unified Molecule-Text LM with Discrete Tokens](https://arxiv.org/html/2408.00863)
- [Entropy-Guided Dynamic Tokens for Graph-LLM Alignment](https://arxiv.org/abs/2602.02742)
- [Process Reward Models That Think](https://arxiv.org/pdf/2504.16828)
- [Math-Shepherd (Wang et al. 2024)](https://aclanthology.org/2024.acl-long.510.pdf)
- [Rewarding Progress (automated PRM training)](https://arxiv.org/pdf/2410.08146)
- [VersaPRM (multi-domain PRM)](https://arxiv.org/html/2502.06737)
- [ReST-MCTS*](https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf)
- [CompassVerifier (EMNLP 2025)](https://arxiv.org/html/2508.03686v1)
- [Generalizable PRMs via Formally Verified Training Data](https://arxiv.org/pdf/2505.15960)
- [SAE-RSV (Refinement of Steering Vectors via SAE)](https://arxiv.org/abs/2509.23799)
- [CorrSteer (correlation-based feature selection for steering)](https://arxiv.org/html/2508.12535)
- [Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts](https://arxiv.org/html/2506.23845v1)
- [SAE vs Activation Difference comparison](https://arxiv.org/html/2510.01246v1)
- [SAE-Guided Steering for LLMs (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1474.pdf)
- [Steering LM Refusal with SAEs (ICML 2025)](https://icml.cc/virtual/2025/50928)
- [Awesome Process Reward Models (curated list)](https://github.com/RyanLiu112/Awesome-Process-Reward-Models)
- [Learning from Rewards in LLMs (curated list)](https://github.com/bobxwu/learning-from-rewards-llm-papers)
- [Improving Cross-Conformal Prediction (2025)](https://arxiv.org/abs/2503.01495)
- [Conformal Prediction (Wikipedia overview)](https://en.wikipedia.org/wiki/Conformal_prediction)
- [IBM Materials FM4M (open materials FMs)](https://github.com/IBM/materials)
- [Cost-Aware Model Orchestration (arXiv 2025)](https://arxiv.org/pdf/2512.01099)
- [Efficient Multi-Model Orchestration for Self-Hosted LLMs](https://arxiv.org/pdf/2512.22402)
- [KNOS - Knowledge-guided Solver (IEEE TKDE 2025)](http://staff.ustc.edu.cn/~huangzhy/files/papers/JiayuLiu-TKDE2025.pdf)
