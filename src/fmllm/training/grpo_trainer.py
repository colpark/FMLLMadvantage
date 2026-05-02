"""GRPO fine-tuning of Llama 3.1 8B on verifier-driven rewards.

Uses ``trl.GRPOTrainer`` with LoRA adapters via ``peft``. The reward
function comes from
:func:`fmllm.training.reward.make_verifier_reward_fn`: each
generated completion is parsed for actions, the actions are
replayed against the FM runners, and the verifier's verdict on the
final commit drives the score.

The trainer generates fresh completions per prompt every step. The
reward function therefore re-runs the verifier (and any FM the
completion calls) on every completion, which is the inner-loop
cost of online RL. Use a small batch size and a modest number of
generations per prompt to keep wall clock manageable.

Depends on:
    transformers, trl, datasets, peft (lazy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fmllm.training.lora import apply_lora, save_lora
from fmllm.training.reward import RewardFn


def train_grpo(
    *,
    base_model_name: str,
    grpo_prompts: list[dict[str, Any]],
    reward_fn: RewardFn,
    output_dir: Path | str,
    learning_rate: float = 5.0e-6,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    num_generations: int = 4,
    max_prompt_length: int = 2048,
    max_completion_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    kl_beta: float = 0.04,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    seed: int = 0,
    bf16: bool = True,
) -> Path:
    """Run GRPO fine-tuning. Saves the LoRA adapter to
    ``output_dir/adapter/`` and returns ``output_dir``.

    ``grpo_prompts`` is the dict list produced by
    :func:`fmllm.training.dataset.trajectories_to_grpo_prompts`.
    """
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
    )
    model = apply_lora(
        base_model,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    )

    def render_prompt(record: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": text}

    ds = Dataset.from_list(grpo_prompts).map(
        render_prompt,
        remove_columns=["messages"],
    )

    config = GRPOConfig(
        output_dir=str(output_dir / "trainer"),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        beta=kl_beta,
        bf16=bf16,
        seed=seed,
        report_to=[],
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )
    trainer.train()

    save_lora(model, output_dir / "adapter")
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    return output_dir


__all__ = ["train_grpo"]
