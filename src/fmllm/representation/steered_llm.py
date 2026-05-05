"""LLM wrapper that applies SAE activation steering during inference.

Phase 15 Stage D. Wraps any object with a ``chat(messages) -> str``
method (typically :class:`fmllm.orchestrator.llm.TransformersLLM`)
so that every forward pass during ``chat()`` has an
:class:`ActivationSteerer` hook attached at a specified residual-
stream layer. The hook adds ``coefficient * decoder_column[fid]``
to the residual at the hooked layer, broadcasting across every
token position in the generation.

This is the canonical Templeton et al. 2024 / Golden Gate Claude
recipe applied to OHVD: every Qwen turn during the loop sees the
same constant bias in its hidden state, so the steering effect is
applied uniformly across the multi-step trajectory.

Usage::

    base_llm = TransformersLLM(model_name="Qwen/Qwen2.5-7B-Instruct")
    steered = SteeredLLMWrapper(
        llm=base_llm,
        sae_dir=Path("checkpoints/qwen_sae/<run_id>"),
        feature_idx=8421,
        coefficient=-2.0,             # ablate the wrong-PASS feature
        layer_path="model.layers.14", # must match the SAE's training layer
    )
    # Then pass `steered` anywhere `base_llm` would have been used.
    response = steered.chat(messages)

Depends on:
    torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fmllm.representation.llm_sae import (
    ActivationSteerer,
    resolve_layer_module,
)


class SteeredLLMWrapper:
    """Wrap any chat-style LLM with SAE activation steering.

    The wrapper defers SAE loading until the first ``chat()`` call
    so that the underlying LLM has a chance to materialize its
    weights (TransformersLLM defers via ``_ensure_loaded()``). We
    also resolve the target ``nn.Module`` layer once on the first
    chat, which works for both plain HF models and PEFT-wrapped
    ones via the ``peft_config`` discriminator.
    """

    def __init__(
        self,
        *,
        llm: Any,
        sae_dir: Path,
        feature_idx: int,
        coefficient: float = 1.0,
        layer_path: str = "model.layers.14",
    ) -> None:
        self._llm = llm
        self._sae_dir = Path(sae_dir)
        self._feature_idx = int(feature_idx)
        self._coefficient = float(coefficient)
        self._layer_path = str(layer_path)
        self._direction: torch.Tensor | None = None
        self._layer_module = None

    @property
    def coefficient(self) -> float:
        return self._coefficient

    @coefficient.setter
    def coefficient(self, value: float) -> None:
        """Allow runtime adjustment of the steering strength."""
        self._coefficient = float(value)

    def _underlying_model(self) -> torch.nn.Module:
        """Return the loaded HF model behind the LLM wrapper.

        TransformersLLM keeps the actual model at ``self._model``
        after ``_ensure_loaded()``. We call into that path so the
        steering hook resolves the right object.
        """
        if hasattr(self._llm, "_ensure_loaded"):
            self._llm._ensure_loaded()
        if hasattr(self._llm, "_model") and self._llm._model is not None:
            return self._llm._model
        raise RuntimeError(
            "SteeredLLMWrapper expected the underlying LLM to expose "
            "._model after _ensure_loaded(); got an object of type "
            f"{type(self._llm).__name__}."
        )

    def _maybe_load_steering(self) -> None:
        if self._direction is not None and self._layer_module is not None:
            return
        model = self._underlying_model()
        sae_path = self._sae_dir / "sae.pt"
        if not sae_path.exists():
            raise FileNotFoundError(f"no sae.pt under {self._sae_dir}")
        payload = torch.load(sae_path, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        if "decoder.weight" not in state:
            raise KeyError(
                "SAE checkpoint is missing decoder.weight; cannot extract "
                "the feature direction."
            )
        decoder_weight = state["decoder.weight"]                # (in_dim, hidden_dim)
        if self._feature_idx < 0 or self._feature_idx >= decoder_weight.shape[1]:
            raise IndexError(
                f"feature_idx={self._feature_idx} out of range "
                f"[0, {decoder_weight.shape[1]})"
            )
        direction = decoder_weight[:, self._feature_idx].detach().clone()

        # Resolve target dtype/device from the LM. For 4-bit models
        # parameters are mixed; bf16 is a safe additive dtype.
        try:
            target_device = next(model.parameters()).device
        except StopIteration:                                    # pragma: no cover
            target_device = torch.device("cpu")
        target_dtype = torch.bfloat16
        for p in model.parameters():
            if p.dtype in (torch.float16, torch.bfloat16, torch.float32):
                target_dtype = p.dtype
                break

        self._direction = direction.to(target_device, dtype=target_dtype)

        # PEFT detection: same logic as the harvester. Plain HF models
        # also have a `.base_model` shortcut, so we test for peft_config.
        is_peft = hasattr(model, "peft_config") and bool(
            getattr(model, "peft_config", None)
        )
        base = model.base_model.model if is_peft else model
        self._layer_module = resolve_layer_module(base, self._layer_path)

    # ------------------------------------------------------------------
    # Pass-through chat with steering hook attached.
    # ------------------------------------------------------------------
    def chat(self, messages: list[dict[str, str]]) -> str:
        self._maybe_load_steering()
        assert self._direction is not None
        assert self._layer_module is not None
        with ActivationSteerer(
            layer_module=self._layer_module,
            feature_direction=self._direction,
            coefficient=self._coefficient,
        ):
            return self._llm.chat(messages)


__all__ = ["SteeredLLMWrapper"]
