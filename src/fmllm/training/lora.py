"""LoRA helpers for parameter-efficient fine-tuning.

Wraps :mod:`peft` so trainers can build, save, and load LoRA
adapters with a small surface. The functions lazy-import ``peft``
so importing this module on a host without the dependency does not
fail.

Default LoRA target modules cover the standard set for Llama-style
decoder transformers: ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``,
``gate_proj``, ``up_proj``, ``down_proj``. Override per architecture
via :func:`apply_lora(target_modules=...)`.

Depends on:
    peft (lazy).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def apply_lora(
    model: Any,
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    bias: str = "none",
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Wrap an HF causal-LM model with a LoRA adapter and return it.

    Args:
        model: A ``transformers.AutoModelForCausalLM`` instance.
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout.
        bias: peft bias mode (``none``, ``all``, or ``lora_only``).
        target_modules: Which projection layers to wrap.
        task_type: peft task type. ``"CAUSAL_LM"`` for our use.

    Returns:
        The wrapped (PEFT) model with LoRA adapters trainable and
        the base weights frozen.
    """
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=list(target_modules),
        task_type=getattr(TaskType, task_type),
    )
    return get_peft_model(model, config)


def save_lora(model: Any, output_dir: Path | str) -> Path:
    """Save the PEFT adapter weights to ``output_dir``.

    The saved directory contains the LoRA tensors and a small config
    JSON. The base model is not duplicated; load via
    :func:`load_lora`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if save_pretrained is None:
        raise TypeError(
            "model has no save_pretrained method; "
            "ensure apply_lora was called before save_lora"
        )
    save_pretrained(str(output_dir))
    return output_dir


def load_lora(
    base_model: Any,
    adapter_dir: Path | str,
    *,
    is_trainable: bool = False,
) -> Any:
    """Attach a saved LoRA adapter onto ``base_model`` and return it."""
    from peft import PeftModel  # noqa: PLC0415

    return PeftModel.from_pretrained(
        base_model, str(adapter_dir), is_trainable=is_trainable,
    )


def lora_parameter_summary(model: Any) -> dict[str, int]:
    """Return a count of trainable vs total parameters."""
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"trainable": trainable, "total": total}


__all__ = [
    "DEFAULT_TARGET_MODULES",
    "apply_lora",
    "load_lora",
    "lora_parameter_summary",
    "save_lora",
]
