"""SFT (supervised fine-tuning) on verifier-passing trajectories.

The simplest Pipeline B variant: behavioral cloning of the LLM's own
verifier-passing trajectories. Useful as a baseline against GRPO and
DPO. This trainer applies the chat template to each record's
``messages`` and minimizes next-token cross-entropy.

Depends on:
    transformers, datasets, peft (lazy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fmllm.training.lora import apply_lora, save_lora


def train_sft(
    *,
    base_model_name: str,
    sft_records: list[dict[str, Any]],
    output_dir: Path | str,
    learning_rate: float = 1.0e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    warmup_steps: int = 50,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    seed: int = 0,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
) -> Path:
    """Run SFT on the (messages,) records produced by
    :func:`fmllm.training.dataset.trajectories_to_sft_records`.

    Saves the LoRA adapter to ``output_dir/adapter/``. Returns the
    output directory.

    Lazy-imports transformers and datasets so this module loads
    cleanly without those packages installed locally.
    """
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

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
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # Gradient checkpointing keeps activation memory bounded by
    # recomputing during backward. Required for 7B-class models on a
    # single 80 GB H100 with reasonable sequence lengths.
    if gradient_checkpointing:
        # Disable use_cache to silence the deprecation warning + ensure
        # checkpointing works.
        if hasattr(model, "config"):
            model.config.use_cache = False
        # PEFT-wrapped models need this so input embeddings receive
        # gradients during the backward pass through checkpointed
        # blocks.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    def render(record: dict[str, Any]) -> dict[str, list[int]]:
        # Two-step: format as a single string with the chat template,
        # then tokenize. This is robust across transformers versions
        # where ``apply_chat_template(tokenize=True)`` may return a
        # tensor, list, dict, or even a formatted string depending on
        # the tokenizer's internal handling.
        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(text, str):
            text = str(text)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
            return_tensors=None,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    ds = Dataset.from_list(sft_records).map(
        render,
        remove_columns=list(sft_records[0].keys()),
    )

    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        bf16=bf16,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        # Bound memory growth from peak allocations on Hopper.
        optim="adamw_torch",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    save_lora(model, output_dir / "adapter")
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    return output_dir


__all__ = ["train_sft"]
