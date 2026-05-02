"""DPO (Direct Preference Optimization) fallback for Pipeline B.

When GRPO does not converge cleanly (the verifier signal is too
sparse, or the inner-loop cost is prohibitive), DPO offers a more
stable alternative. The trainer uses (passing trajectory, failing
trajectory) preference pairs from the same specimen as direct
supervision: maximize the LLM's likelihood ratio between chosen and
rejected completions.

Pair shape (from
:func:`fmllm.training.dataset.trajectories_to_dpo_pairs`):

    {
      "specimen_id": int,
      "query": str,
      "prompt_messages": [...],   # system + user
      "chosen": str,              # concatenated assistant turns from a PASS trajectory
      "rejected": str,            # concatenated assistant turns from a FAIL trajectory
    }

Depends on:
    transformers, trl, datasets, peft (lazy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fmllm.training.lora import apply_lora, save_lora


def train_dpo(
    *,
    base_model_name: str,
    dpo_pairs: list[dict[str, Any]],
    output_dir: Path | str,
    learning_rate: float = 5.0e-6,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_prompt_length: int = 2048,
    max_length: int = 4096,
    beta: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    seed: int = 0,
    bf16: bool = True,
) -> Path:
    """Run DPO fine-tuning on preference pairs. Saves LoRA adapter to
    ``output_dir/adapter/``."""
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

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
        base_model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    )

    def render(record: dict[str, Any]) -> dict[str, str]:
        prompt = tokenizer.apply_chat_template(
            record["prompt_messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }

    ds = Dataset.from_list(dpo_pairs).map(
        render,
        remove_columns=[
            c for c in ("messages", "prompt_messages", "specimen_id", "query")
            if c in dpo_pairs[0]
        ],
    )

    config = DPOConfig(
        output_dir=str(output_dir / "trainer"),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        beta=beta,
        bf16=bf16,
        seed=seed,
        report_to=[],
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()

    save_lora(model, output_dir / "adapter")
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    return output_dir


__all__ = ["train_dpo"]
