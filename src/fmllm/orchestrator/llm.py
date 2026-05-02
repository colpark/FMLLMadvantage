"""LLM wrappers for the OHVD loop.

Three pieces:

    - :class:`BaseLLM`: the abstract interface the loop consumes. The
      loop holds a list of chat messages and asks the LLM for one new
      assistant turn per iteration.
    - :class:`MockLLM`: returns scripted responses for tests. Local
      pytest hits this exclusively.
    - :class:`TransformersLLM`: wraps Llama 3.1 8B Instruct (or any
      compatible chat model) through ``transformers``. Used on the
      remote at inference time. The class lazy-imports torch /
      transformers so the module loads cleanly on a laptop without GPU
      libraries.

Action protocol:
    The loop instructs the model via system prompt to emit ONE JSON
    action per turn, with one of three shapes:

        {"action": "call_fm", "tool_name": "fm1", "specimen_id": 42}
        {"action": "hypothesize", "claim": {"n_atoms": 7, ...}}
        {"action": "commit", "claim": {"n_atoms": 7, ...}}

    :func:`parse_llm_response` extracts the first JSON object in the
    response, validates it against :class:`LLMAction`, and returns
    the typed action. Malformed responses produce
    ``LLMAction(action_type=ERROR, error=...)`` so the loop logs and
    proceeds rather than crashing.

Depends on:
    pydantic.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from fmllm.orchestrator.trajectory import (
    ActionType,
    LLMAction,
    ToolCall,
)
from fmllm.verifier.schema import PhysicalStateClaim


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_response(text: str) -> LLMAction:
    """Extract the first JSON object from ``text`` and validate it."""
    if not text or not text.strip():
        return LLMAction(
            action_type=ActionType.ERROR,
            error="empty response",
            raw_text=text or "",
        )

    match = _JSON_RE.search(text)
    if match is None:
        return LLMAction(
            action_type=ActionType.ERROR,
            error="no JSON object found",
            raw_text=text,
        )

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return LLMAction(
            action_type=ActionType.ERROR,
            error=f"json parse error: {exc.msg}",
            raw_text=text,
        )
    if not isinstance(payload, dict):
        return LLMAction(
            action_type=ActionType.ERROR,
            error="JSON root is not an object",
            raw_text=text,
        )

    action = payload.get("action")
    try:
        if action == "call_fm":
            tool_name = payload.get("tool_name")
            args = {k: v for k, v in payload.items() if k not in {"action", "tool_name"}}
            if not isinstance(tool_name, str):
                raise ValueError("call_fm requires string tool_name")
            return LLMAction(
                action_type=ActionType.CALL_FM,
                tool_call=ToolCall(tool_name=tool_name, arguments=args),
                raw_text=text,
            )
        if action in ("hypothesize", "commit"):
            claim_payload = payload.get("claim")
            if not isinstance(claim_payload, dict):
                raise ValueError(f"{action} requires a dict 'claim' field")
            claim = PhysicalStateClaim.model_validate(claim_payload)
            return LLMAction(
                action_type=(
                    ActionType.COMMIT if action == "commit" else ActionType.HYPOTHESIZE
                ),
                claim=claim,
                raw_text=text,
            )
        return LLMAction(
            action_type=ActionType.ERROR,
            error=f"unknown action {action!r}; expected one of "
            "call_fm, hypothesize, commit",
            raw_text=text,
        )
    except Exception as exc:  # noqa: BLE001
        return LLMAction(
            action_type=ActionType.ERROR,
            error=f"action validation error: {exc!s}",
            raw_text=text,
        )


class BaseLLM(ABC):
    """Abstract chat LLM the orchestrator loop consumes."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate one assistant turn from the conversation messages.

        Returns the raw model output. The loop calls
        :func:`parse_llm_response` on the result.
        """


class MockLLM(BaseLLM):
    """Deterministic mock LLM that emits scripted responses.

    Pass a list of strings (each representing one assistant turn) at
    construction time. Two exhaustion modes:

    - ``cycle=False`` (default): the mock returns each response in
      order. Once the script runs out, it returns
      ``{"action": "error", "error": "mock exhausted"}`` so the loop
      terminates cleanly. Use this for single-specimen smoke tests.
    - ``cycle=True``: the mock loops the script indefinitely. Each
      specimen in a batch collection therefore replays the same
      action sequence from the start.
    """

    def __init__(
        self,
        scripted_responses: list[str],
        *,
        cycle: bool = False,
    ) -> None:
        self._responses: list[str] = list(scripted_responses)
        self._cycle: bool = bool(cycle)
        self._cursor: int = 0
        self.call_count = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        self.received_messages.append([dict(m) for m in messages])
        if not self._responses:
            return json.dumps({"action": "error", "error": "mock script empty"})
        if self._cycle:
            response = self._responses[self._cursor % len(self._responses)]
            self._cursor += 1
            return response
        if self._cursor >= len(self._responses):
            return json.dumps({"action": "error", "error": "mock exhausted"})
        response = self._responses[self._cursor]
        self._cursor += 1
        return response


class TransformersLLM(BaseLLM):
    """Llama 3.1 8B Instruct (or compatible) wrapper via ``transformers``.

    The class loads the model lazily at first ``chat`` call so importing
    this module on a CPU-only laptop does not trigger the heavy
    transformers / torch dependencies until an actual inference run.

    Args:
        model_name: HuggingFace model id (default
            ``meta-llama/Llama-3.1-8B-Instruct``).
        device: Either ``"auto"``, ``"cuda"``, ``"cpu"``, or a specific
            ``"cuda:N"``.
        max_new_tokens: Maximum tokens generated per chat call.
        temperature: Sampling temperature. ``0.0`` triggers greedy
            decoding.
        dtype: Optional override; defaults to ``torch.bfloat16`` on
            CUDA, ``torch.float32`` on CPU.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        *,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        dtype: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM, AutoTokenizer,
        )

        dtype = self.dtype
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = self.device if self.device != "auto" else "auto"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def chat(self, messages: list[dict[str, str]]) -> str:
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        assert self._tokenizer is not None and self._model is not None
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            do_sample = self.temperature > 0.0
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else 1.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen = output[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)


__all__ = [
    "BaseLLM",
    "MockLLM",
    "TransformersLLM",
    "parse_llm_response",
]
