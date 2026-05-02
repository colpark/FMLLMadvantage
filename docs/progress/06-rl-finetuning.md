# Phase 6: RL Fine-Tuning (Pipeline B)

## What I built

The Pipeline B training stack collects verifier-driven trajectories
from Pipeline A and uses them to fine-tune the orchestrator LLM with
LoRA adapters. Three trainer flavors share the data pipeline.

### `fmllm.training`

- `trajectory_collection.py` - `collect_trajectories(...)` runs the
  OHVD loop on a list of specimens and writes JSONL plus a summary.
  `load_trajectories_jsonl` and `write_trajectories_jsonl` round-trip
  the file. `filter_passing=True` drops non-PASS rows on write.
- `dataset.py` - converts trajectories to trainer-shaped records:
  - `trajectory_to_messages` rebuilds the chat history (system,
    user, assistant turns, tool messages) the loop fed the LLM at
    runtime.
  - `trajectories_to_sft_records` produces (messages, ...) for SFT.
  - `trajectories_to_dpo_pairs` produces preference pairs sharing
    specimen + query, with chosen=PASS and rejected=FAIL.
  - `trajectories_to_grpo_prompts` produces deduplicated prompts.
- `reward.py` - `make_verifier_reward_fn(verifier, runners, ...)`
  builds the GRPO reward function. The function uses a
  brace-balanced JSON scanner to extract every action from the
  generated completion (handling nested ``claim`` dicts), replays
  the FM calls, and runs the verifier on the final commit.
  Reward: +1.0 PASS / +0.3 CAVEAT / +0.05 per source PASS.
- `lora.py` - `apply_lora(model)` wraps an HF causal-LM with peft
  LoRA adapters. Default target modules cover Llama attention +
  MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`). `save_lora` and `load_lora`
  round-trip the adapter weights.
- `sft_trainer.py` - SFT via `transformers.Trainer`. Behavioral
  cloning of verifier-passing trajectories.
- `grpo_trainer.py` - GRPO via `trl.GRPOTrainer`. Online RL with
  the verifier reward; the trainer generates new completions per
  prompt every step, the reward function runs the verifier on the
  parsed completion.
- `dpo_alternative.py` - DPO via `trl.DPOTrainer`. Use when GRPO
  does not converge cleanly.

### Tests (`tests/test_training.py`, 13 passing + 1 skipped, 174 total)

- Trajectory collection writes JSONL + summary; round-trip preserves.
- `filter_passing=True` drops failed trajectories.
- `trajectory_to_messages` reconstructs the chat history.
- SFT / DPO / GRPO dataset builders behave correctly (`only_passing`
  filter, PASS+FAIL pair construction, GRPO prompt deduplication).
- `extract_specimen_id` parses the user message.
- Reward function returns 0 without a commit, positive for a
  reasonable claim, and 0 under V0 ablation (verifier all SKIP).
- LoRA save/load round-trip on tiny GPT-2 (gated by `importorskip`
  so local hosts without `peft` skip the test).

### CLIs

- `scripts/collect_trajectories.py` - run Pipeline A on a range of
  specimens, write JSONL.
- `scripts/train_pipeline_b.py` - unified Pipeline B trainer with
  `--mode {sft, dpo, grpo}`.
- `scripts/collect_trajectories.sh` - bash wrapper. Defaults to
  the mock LLM. Pass `--real` as the first arg to use Llama 3.1.
- `scripts/train_pipeline_b.sh` - bash wrapper. Picks the latest
  `trajectories.jsonl` under `runs/trajectories/` unless
  `TRAJECTORIES` env overrides.

## What the user runs to verify Phase 6

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest -m "not gpu" -v
```

Expect 174 passing tests + 1 skipped (peft).

### Remote 4xH100 host

#### Step 1. Collect trajectories with the mock LLM (~minutes)

```
bash scripts/collect_trajectories.sh 0 200 train_50k
```

This uses the scripted MockLLM (no auth needed, no LLM weights
loaded) and produces a JSONL with 200 trajectories under
`runs/trajectories/<run_id>/`. Most will end CAVEAT or FAIL with
the static mock script; the JSONL still serves as a smoke test of
the collection pipeline.

#### Step 2. Collect trajectories with the real LLM (~hour for ~500 specimens)

```
huggingface-cli login   # one-time, only if Llama is gated for you
bash scripts/collect_trajectories.sh --real 0 500 train_50k
```

Expected:
- First call loads the LLM weights (~16 GB, a few minutes).
- Per-specimen wall clock: 30 to 90 seconds (depends on step
  budget and how often the LLM commits early).
- Disk: ~10-50 KB per trajectory; 500 trajectories ~5-20 MB.

#### Step 3. Train Pipeline B

GPU memory and runtime estimates per mode for Llama 3.1 8B + LoRA
on a single H100 80 GB:

| Mode | GPU mem | Wall clock (1 epoch on ~500 PASS records) |
|------|---------|-------------------------------------------|
| SFT  | ~24 GB  | ~30 min                                   |
| DPO  | ~32 GB  | ~45 min                                   |
| GRPO | ~40 GB  | ~3-6 hours (inner-loop verifier dominant) |

Multi-GPU via `accelerate launch` cuts SFT and DPO roughly
linearly. GRPO scales sublinearly because each rollout serializes
on the verifier.

```
# SFT (cleanest baseline)
bash scripts/train_pipeline_b.sh sft

# DPO (preference pairs)
bash scripts/train_pipeline_b.sh dpo

# GRPO (online RL)
ACCELERATE=1 bash scripts/train_pipeline_b.sh grpo
```

Each run writes the LoRA adapter to
`checkpoints/pipeline-b/<run_id>/adapter/`.

## Convergence indicators

The training scripts log:

- **SFT**: cross-entropy loss per logging step. Expect a steady
  drop to under 1.0 within the first epoch on small datasets.
- **DPO**: chosen / rejected reward gap and policy KL. The chosen
  reward should rise relative to rejected within hundreds of
  steps; KL should stay below ~0.5 until late training.
- **GRPO**: mean reward, KL to reference policy, and reward std.
  Mean reward should trend upward; std too small early on means
  the policy collapsed; KL above ~1.0 means the policy drifted
  too far from the reference.

## Known caveats

- **GRPO inner-loop cost.** The reward function calls every FM in
  ``runners`` and the full verifier per generated completion. With
  4 generations per prompt and a 100-prompt epoch, that's 400
  reward evaluations, each taking ~1-2 seconds on H100 (FM
  forward passes dominate). Plan GRPO runs accordingly.
- **DPO needs both PASS and FAIL.** Specimens that always commit
  to the same outcome (typical for the static mock) produce no
  preference pairs. Run trajectory collection with the real LLM
  at a higher temperature to mix outcomes.
- **SFT is the safest baseline.** If the trajectories don't yield
  enough PASS records, SFT yields a tighter LLM that mimics the
  base policy on the cases the verifier likes. Use this as the
  reference Pipeline B run.
- **No real-LLM integration test.** The trainer wrappers ship
  untested at the model-loading layer; the first remote run is the
  integration test. Use a tiny base model (e.g.,
  `microsoft/Phi-3.5-mini-instruct`) for a quick wiring check
  before committing to Llama 3.1 8B.

## What remains for Phase 7

- Implement `src/fmllm/evaluation/`: the eight world-model tests.
  Trajectory-level (compression, distinction, step recoverability),
  prediction-level (compression, distinction, goal competence),
  cross-layer (federated factorability, calibrated uncertainty).
- `scripts/run_evaluation.py` CLI.
