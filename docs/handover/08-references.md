# 08 — References

## External literature directly cited or relied on

### SAE on LLMs

- **Bricken et al., "Towards Monosemanticity: Decomposing
  Language Models with Dictionary Learning"** (Anthropic, 2023).
  First major demonstration that SAEs trained on transformer
  activations decompose dense polysemantic neurons into sparse
  monosemantic features. <https://transformer-circuits.pub/2023/monosemantic-features/index.html>

- **Templeton et al., "Scaling Monosemanticity: Extracting
  Interpretable Features from Claude 3 Sonnet"** (Anthropic,
  2024). Scaled SAE to a production LLM with ~16M features.
  Identified safety-relevant features (sycophancy, deception,
  the famous "Golden Gate Bridge"). The intellectual basis for
  Phase 15. <https://transformer-circuits.pub/2024/scaling-monosemanticity/>

- **Anthropic blog, "Golden Gate Claude"** (2024). Steering demo
  where the Golden-Gate-Bridge feature was clamped at
  ~10x activation; the model became obsessed with the bridge.
  Canonical activation-steering example. <https://www.anthropic.com/news/golden-gate-claude>

- **Gao et al., "Scaling and evaluating sparse autoencoders"**
  (OpenAI, 2024). Top-K SAE recipe — what we use throughout
  Phase 13 and Phase 15. <https://openai.com/index/extracting-concepts-from-gpt-4/>

- **Cunningham et al., "Sparse Autoencoders Find Highly
  Interpretable Features in Language Models"** (2023). Early
  demonstration on small models.

- **Lindsey et al., "On the Biology of a Large Language Model"**
  / "Tracing the thoughts of a large language model" (Anthropic,
  2025). SAE features as nodes in attribution graphs; circuits
  for multi-step reasoning. <https://transformer-circuits.pub/2025/attribution-graphs/biology.html>

### Causal abstraction / interpretability

- **Geiger et al., "Causal Abstractions of Neural Networks"**
  (NeurIPS 2021). Formal framework for when one network is a
  causal abstraction of another. Foundation for Bet 2.

- **Wu et al., "Interpretability at Scale: Identifying Causal
  Mechanisms in Alpaca"** (ICML 2024). Distributed alignment
  search — interchange interventions to find causally-relevant
  representations.

- **Vig et al., "Investigating Gender Bias in Language Models
  Using Causal Mediation Analysis"** (NeurIPS 2020). Mediation
  analysis methodology that informs Phase 14's causal-audit
  approach.

### Tool use and agentic LLMs

- **Schick et al., "Toolformer: Language Models Can Teach
  Themselves to Use Tools"** (NeurIPS 2023). The
  trained-tool-use paradigm.

- **Yao et al., "ReAct: Synergizing Reasoning and Acting in
  Language Models"** (ICLR 2023). The interleaved-reasoning-
  and-acting pattern that the OHVD loop generalizes.

- **Anthropic, "Claude with computer use"** (2024). End-to-end
  agentic LLM trained on tool use; relevant to Bet 1's
  scaling story.

- **OpenAI, "Learning to reason with LLMs"** (o1 release, 2024).
  Inference-time CoT scaling; relevant to whether Phase 11 Stages
  3-4 might close the gap.

### CoT and reasoning fine-tuning

- **Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in
  Large Language Models"** (NeurIPS 2022). Original CoT.

- **Zelikman et al., "STaR: Bootstrapping Reasoning With
  Reasoning"** (NeurIPS 2022). Rejection-sampling on correct
  CoTs; the unbuilt Stage 3 of Phase 11.

- **Shao et al., "DeepSeekMath: Pushing the Limits of
  Mathematical Reasoning in Open Language Models"** (2024). GRPO
  formulation; the framework Phase 11 Stage 4 would use.

### LoRA and fine-tuning

- **Hu et al., "LoRA: Low-Rank Adaptation of Large Language
  Models"** (ICLR 2022). The fine-tuning method used in Phase 11
  Stage 2.

- **Dettmers et al., "QLoRA: Efficient Finetuning of Quantized
  LLMs"** (NeurIPS 2023). 4-bit quantization (nf4) we use for
  inference.

### Conformal prediction

- **Angelopoulos and Bates, "A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification"**
  (2021). Foundation of the conformal verifier source.

### Models and tokenizers used

- **Yang et al., "Qwen2.5 Technical Report"** (2024). The LLM we
  used.

- **Salesforce BLIP image captioning** (Li et al., 2022). Used
  in `naked_vision` baseline as a generic VLM caption.

### Lennard-Jones / cluster physics (testbed grounding)

- **Doye, Wales et al.**, lattice cluster databases. Source of
  the literature DB references our verifier checks against.

## Internal documentation cross-references

These are the original chronological narrative; the handover
folder summarizes them.

| Doc | Phase | Status |
|---|---|---|
| `docs/progress/00-init.md` | 0 (init) | reference |
| `docs/progress/01-testbed.md` | 1 (testbed) | reference |
| `docs/progress/02-fms.md` | 2 (FMs) | reference |
| `docs/progress/03-bridges.md` | 3 (bridges) | reference |
| `docs/progress/04-verifier.md` | 4 (verifier + V0..V4 ablation) | reference |
| `docs/progress/05-orchestration.md` | 5 (OHVD loop) | reference |
| `docs/progress/06-rl-finetuning.md` | 6 (RL infra) | reference |
| `docs/progress/07-evaluation.md` | 7 (eight tests) | reference |
| `docs/progress/08-baselines.md` | 8a (baselines) | reference |
| `docs/progress/09-connector.md` | 9 (Layer C) | closed neg |
| `docs/progress/10-ssl-fm2.md` | 10 (Layer D) | closed neg |
| `docs/progress/11-cot-sft.md` | 11 (CoT-SFT) | closed neg |
| `docs/progress/13-sae.md` | 13 (FM2 SAE) | closed neg |
| `docs/progress/14-sae-causal.md` | 14 (causal audit) | closed neg |
| `docs/progress/15-qwen-sae.md` | 15 (Qwen SAE) | closed neg |

Audits in `docs/audits/<phase>-audit.md` follow each closed phase
with the reproduction protocol.

## Code references for each method

| Method | Primary file | Tests |
|---|---|---|
| TopKSAE | `src/fmllm/representation/sae.py` | `tests/test_sae.py` |
| FM SAE labels | `src/fmllm/representation/labels.py` | `tests/test_sae.py` |
| Causal interventions | `src/fmllm/representation/causal.py` | `tests/test_sae_causal.py` |
| LLM hooks | `src/fmllm/representation/llm_sae.py` | `tests/test_llm_sae.py` |
| LLM SAE labels | `src/fmllm/representation/llm_labels.py` | `tests/test_llm_labels.py` |
| Steered LLM wrapper | `src/fmllm/representation/steered_llm.py` | `tests/test_steered_llm.py` |
| Probe bank + synthetic CoT | `src/fmllm/training/probe_bank.py`, `synthetic_cot.py` | `tests/test_synthetic_cot.py` |
| OHVD loop | `src/fmllm/orchestrator/loop.py` | `tests/test_orchestration.py` |
| Verifier (V0..V4) | `src/fmllm/verifier/` | `tests/test_verifier*.py` |

## Datasets

- `data/synthetic_lj_v1/specimens.h5` — 50K + 200 specimens with
  ground-truth (motif_id, atom_count, temperature, energy,
  positions, RDF).
- `data/synthetic_lj_v1/splits.yaml` — train/test split
  definitions including `train_50k` and the held-out range.
- `data/literature/clusters.json` — known-cluster reference DB
  used by the literature verifier source.

## External tooling

- **`uv`** package manager (Astral) — pinned at 0.10.x.
- **HuggingFace `transformers`** — for Qwen loading and chat
  template handling.
- **HuggingFace `peft`** — for LoRA adapter loading and stacking.
- **`bitsandbytes`** — for 4-bit nf4 quantization.
- **`h5py`** — for the synthetic dataset.
- **`pydantic`** — for the typed schemas (BridgedFMOutput,
  Trajectory, etc.).
- **`typer`** — for CLI argument parsing.
- **`loguru`** — for structured logging in the orchestrator.

## Where to file findings if you publish

This work intersects:

- ML interpretability (SAE/circuits): NeurIPS, ICLR, ICML
  workshops; the Anthropic / OpenAI / Apollo Research labs'
  open challenge calls.
- LLM agents and tool use: ACL, EMNLP, NeurIPS workshops on
  language agents.
- Scientific machine learning (the testbed framing): NeurIPS
  AI4Science, ICLR ML4PS workshops.

Given the negative-results-as-contribution framing
(`05-architectural-findings.md`), the most natural venue is a
**workshop on negative results in ML** (NeurIPS hosts one
periodically) or a focused workshop on representation reading
between FMs and LLMs.
