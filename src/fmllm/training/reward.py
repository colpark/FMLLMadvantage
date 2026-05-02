"""Reward functions for GRPO fine-tuning.

The reward function takes a list of (prompt, completion) pairs the
trainer's policy produced this step and returns one scalar per pair.
For our project the score derives from the verifier:

    +1.0 for an aggregate PASS verdict on the LLM's commit claim,
    +0.3 for CAVEAT,
    +0.0 for FAIL or SKIP (no commit found),
    +0.05 per source that returned PASS (encourages broad coverage).

The function replays the actions the LLM emitted in the completion
by:

    1. Extracting the specimen ID from the prompt.
    2. Parsing every JSON action block in the completion.
    3. For each ``call_fm`` action: invoking the matching runner.
    4. For the last ``commit`` action: handing the claim to the
       verifier with the bridged outputs collected along the way.

If the completion lacks a commit action, the score is zero.

Depends on:
    fmllm.bridges, fmllm.orchestrator, fmllm.verifier.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from fmllm.fms._schemas import BridgedFMOutput
from fmllm.orchestrator import (
    ActionType,
    parse_llm_response,
)
from fmllm.verifier import (
    PhysicalStateClaim,
    SourceDecision,
    SourcesConfig,
    Verifier,
)


RewardFn = Callable[..., list[float]]


_SPECIMEN_RE = re.compile(r"Specimen id:\s*(\d+)")


def _find_json_objects(text: str) -> list[str]:
    """Return every top-level brace-balanced JSON object in ``text``.

    Stack-based scanner that handles nested objects (so it captures
    ``{"action": "commit", "claim": {...}}`` correctly) and ignores
    braces inside string literals.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
    return out


def extract_specimen_id(prompt: str | list[dict[str, str]]) -> int | None:
    """Recover ``specimen_id`` from the user message of a prompt.

    Accepts either the raw prompt text or a list of chat messages.
    Returns ``None`` when the prompt does not name a specimen.
    """
    if isinstance(prompt, list):
        text = "\n".join(
            m.get("content", "") for m in prompt if m.get("role") in ("user", "system")
        )
    else:
        text = str(prompt)
    m = _SPECIMEN_RE.search(text)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_actions(completion: str) -> list[dict[str, Any]]:
    """Find every JSON action object in a completion string."""
    out: list[dict[str, Any]] = []
    for block in _find_json_objects(completion):
        action = parse_llm_response(block)
        if action.action_type is ActionType.ERROR:
            continue
        out.append({"action": action, "raw": block})
    return out


def make_verifier_reward_fn(
    *,
    verifier: Verifier,
    runners: dict[str, Callable[[dict[str, Any]], BridgedFMOutput]],
    sources_config: SourcesConfig | None = None,
    reward_pass: float = 1.0,
    reward_caveat: float = 0.3,
    reward_per_source_pass: float = 0.05,
) -> RewardFn:
    """Build a TRL-compatible reward function.

    The returned callable accepts the keyword arguments TRL passes
    (``prompts``, ``completions``, plus extras) and returns a list
    of floats.
    """

    def reward_fn(
        completions: list[Any],
        prompts: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        prompts_list: list[Any] = list(prompts or [None] * len(completions))
        rewards: list[float] = []
        for prompt, completion in zip(prompts_list, completions, strict=False):
            score = _score_one(
                prompt=prompt,
                completion=str(completion),
                verifier=verifier,
                runners=runners,
                sources_config=sources_config,
                reward_pass=reward_pass,
                reward_caveat=reward_caveat,
                reward_per_source_pass=reward_per_source_pass,
            )
            rewards.append(score)
        return rewards

    return reward_fn


def _score_one(
    *,
    prompt: Any,
    completion: str,
    verifier: Verifier,
    runners: dict[str, Callable[[dict[str, Any]], BridgedFMOutput]],
    sources_config: SourcesConfig | None,
    reward_pass: float,
    reward_caveat: float,
    reward_per_source_pass: float,
) -> float:
    specimen_id = extract_specimen_id(prompt)
    items = _extract_actions(completion)
    if not items:
        return 0.0

    bridged_outputs: list[BridgedFMOutput] = []
    final_claim: PhysicalStateClaim | None = None

    for entry in items:
        action = entry["action"]
        if action.action_type is ActionType.CALL_FM:
            tool_call = action.tool_call
            if tool_call is None:
                continue
            runner = runners.get(tool_call.tool_name)
            if runner is None:
                continue
            args = dict(tool_call.arguments)
            if "specimen_id" not in args and specimen_id is not None:
                args["specimen_id"] = specimen_id
            try:
                bridged = runner(args)
            except Exception:  # noqa: BLE001
                continue
            bridged_outputs.append(bridged)
        elif action.action_type is ActionType.COMMIT:
            final_claim = action.claim

    if final_claim is None:
        return 0.0

    verdict = verifier.verify(
        bridged_outputs, final_claim, sources_config=sources_config,
    )
    score = 0.0
    decision = verdict.aggregate_decision
    if decision is SourceDecision.PASS:
        score += reward_pass
    elif decision is SourceDecision.CAVEAT:
        score += reward_caveat
    score += reward_per_source_pass * sum(
        1 for v in verdict.source_verdicts if v.decision is SourceDecision.PASS
    )
    return float(score)


__all__ = [
    "RewardFn",
    "extract_specimen_id",
    "make_verifier_reward_fn",
]
