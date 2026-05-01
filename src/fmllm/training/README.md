# fmllm.training

RL fine-tuning utilities (Pipeline B).

Phase 6 will add:
- `trajectory_collection.py` - runs Pipeline A on training queries and
  collects verifier-passing trajectories for RL fine-tuning.
- `grpo_trainer.py` - GRPO-style fine-tuning of the LLM with LoRA
  adapters and a verifier-driven reward.
- `dpo_alternative.py` - DPO-based preference fine-tuning fallback.

Currently empty.
