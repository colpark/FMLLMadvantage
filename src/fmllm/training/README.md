# fmllm.training

Pipeline B: RL fine-tuning of the orchestrator LLM on
verifier-driven signals. Three trainer flavors (SFT, GRPO, DPO)
share a LoRA adapter strategy and lazy-import the heavy stack
(``transformers``, ``trl``, ``peft``, ``datasets``).

## Files

- `trajectory_collection.py` - `collect_trajectories(...)` runs
  Pipeline A across a list of specimens and writes JSONL. Helpers
  `load_trajectories_jsonl` and `write_trajectories_jsonl` round-
  trip the file.
- `dataset.py` - converts trajectories to trainer-shaped records:
  - `trajectory_to_messages` rebuilds the chat history the loop
    showed the LLM at run time.
  - `trajectories_to_sft_records` produces (messages,) for SFT.
  - `trajectories_to_dpo_pairs` produces preference pairs that
    share specimen + query, with chosen=PASS and rejected=FAIL.
  - `trajectories_to_grpo_prompts` produces deduplicated prompt
    records the GRPO trainer samples new completions from.
- `reward.py` - `make_verifier_reward_fn(verifier, runners, ...)`
  builds the GRPO reward function. The function parses the
  generated completion for action-shaped JSON, replays the FM
  calls, and runs the verifier on the final commit claim. Reward:
  +1.0 PASS / +0.3 CAVEAT / +0.05 per source PASS as bonus.
- `lora.py` - `apply_lora(model)` wraps a HF causal-LM with peft
  LoRA adapters. `save_lora` and `load_lora` round-trip the
  adapter weights. Default target modules cover Llama-style
  attention and MLP projections.
- `sft_trainer.py` - `train_sft(...)` runs supervised fine-tuning
  on (messages,) records.
- `grpo_trainer.py` - `train_grpo(...)` runs `trl.GRPOTrainer` with
  the verifier reward. Best for long-horizon online RL.
- `dpo_alternative.py` - `train_dpo(...)` runs `trl.DPOTrainer` on
  preference pairs. Use when GRPO does not converge cleanly.

## Pipeline

```
[Pipeline A on training queries]
         |
         v
[trajectories.jsonl]
   passing + failing
         |
+--------+----------+-----------+
|        |          |           |
v        v          v           v
SFT      DPO        GRPO        (eval)
records  pairs      prompts
   |        |          |
   v        v          v
[Trainer] [Trainer] [Trainer]
   |        |          |
   +---+----+----+-----+
       v
   adapter/
   (LoRA weights)
```

## Reward function

The GRPO reward is the verifier's verdict on the LLM's commit
claim, plus a per-source-pass bonus. The reward function calls
the same `Verifier` instance the orchestrator uses, so the loss
landscape and the inference behavior align.

| Decision | Reward |
|---|---|
| `PASS` | 1.0 |
| `CAVEAT` | 0.3 |
| `FAIL` | 0.0 |
| `SKIP` (no commit found) | 0.0 |
| Bonus | +0.05 per source PASS verdict |

## CLIs

`scripts/collect_trajectories.py` runs the OHVD loop across many
specimens and writes JSONL.

`scripts/train_pipeline_b.py` is the unified Pipeline B trainer.
Pass `--mode {sft, dpo, grpo}` plus the trajectories JSONL.
