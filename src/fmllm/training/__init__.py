"""Pipeline B RL fine-tuning: trajectory collection, datasets, trainers.

The package wraps three trainer flavors (SFT, GRPO, DPO) with LoRA
adapters via ``peft``. Trainers lazy-import ``transformers``,
``datasets``, ``trl``, and ``peft``, so the rest of the project loads
cleanly without those dependencies present.
"""

from fmllm.training.dataset import (
    trajectories_to_dpo_pairs,
    trajectories_to_grpo_prompts,
    trajectories_to_sft_records,
    trajectory_to_messages,
)
from fmllm.training.lora import (
    DEFAULT_TARGET_MODULES,
    apply_lora,
    load_lora,
    lora_parameter_summary,
    save_lora,
)
from fmllm.training.reward import (
    RewardFn,
    extract_specimen_id,
    make_verifier_reward_fn,
)
from fmllm.training.trajectory_collection import (
    collect_trajectories,
    load_trajectories_jsonl,
    write_trajectories_jsonl,
)

__all__ = [
    "DEFAULT_TARGET_MODULES",
    "RewardFn",
    "apply_lora",
    "collect_trajectories",
    "extract_specimen_id",
    "load_lora",
    "load_trajectories_jsonl",
    "lora_parameter_summary",
    "make_verifier_reward_fn",
    "save_lora",
    "trajectories_to_dpo_pairs",
    "trajectories_to_grpo_prompts",
    "trajectories_to_sft_records",
    "trajectory_to_messages",
    "write_trajectories_jsonl",
]
