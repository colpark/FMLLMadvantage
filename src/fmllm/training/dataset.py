"""Convert collected trajectories into trainer-shaped datasets.

Three target shapes:

    SFT  -> (messages,) where messages is the full chat history,
            including the assistant's actions. The trainer applies a
            chat template and computes loss over assistant turns.

    DPO  -> (prompt, chosen, rejected) where ``chosen`` comes from a
            verifier-passing trajectory and ``rejected`` from a
            failing trajectory on the same query / specimen.

    GRPO -> (prompt,) where the trainer generates new completions
            from the prompt and a reward function scores them. Phase 6
            ships :func:`fmllm.training.reward.make_verifier_reward_fn`
            for the reward.

Reconstruction:
    The :class:`Trajectory` records each LLM action's raw text in
    ``Step.llm_action.raw_text`` and the resulting bridged outputs
    or verdicts on subsequent steps with the same ``step_index``.
    :func:`trajectory_to_messages` walks the steps in order and
    rebuilds the chat history the loop showed the LLM.

Depends on:
    pydantic, fmllm.orchestrator.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from fmllm.orchestrator import (
    DEFAULT_SYSTEM_PROMPT,
    StepType,
    Trajectory,
)
from fmllm.orchestrator.loop import (  # private but part of the loop's contract
    _bridged_summary,
    _verdict_summary,
)
from fmllm.verifier import SourceDecision


def trajectory_to_messages(
    traj: Trajectory,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Reconstruct the chat messages the LLM saw plus emitted.

    The function rebuilds the conversation by replaying the
    trajectory's per-step records in order:

        - Each unique ``step_index`` corresponds to one LLM turn.
        - The first step in a group with a populated ``llm_action``
          contributes the assistant message.
        - Subsequent steps in the same group (observation,
          verifier_verdict) contribute tool messages with the same
          summary the runtime loop fed the LLM.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _format_user_message(traj.query, traj.specimen_id),
        },
    ]

    # Walk steps in (step_index, position) order. Within one
    # step_index, the assistant message comes first, then tool
    # messages.
    grouped: dict[int, list] = {}
    for s in traj.steps:
        grouped.setdefault(int(s.step_index), []).append(s)

    for step_idx in sorted(grouped):
        group = grouped[step_idx]
        # Assistant turn from llm_action.raw_text.
        first_with_action = next(
            (s for s in group if s.llm_action is not None), None,
        )
        if first_with_action is not None and first_with_action.llm_action.raw_text:
            messages.append({
                "role": "assistant",
                "content": first_with_action.llm_action.raw_text,
            })
        # Tool messages: observation -> bridged summary; verifier_verdict -> verdict summary.
        for s in group:
            if s.step_type is StepType.OBSERVATION and s.bridged_output is not None:
                messages.append({
                    "role": "tool",
                    "content": json.dumps(_bridged_summary(s.bridged_output)),
                })
            elif s.step_type is StepType.VERIFIER_VERDICT and s.verdict is not None:
                messages.append({
                    "role": "tool",
                    "content": json.dumps(_verdict_summary(s.verdict)),
                })
            elif s.step_type is StepType.ERROR:
                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "error": s.notes or "parse error",
                        "hint": "respond with exactly one JSON action object",
                    }),
                })
    return messages


def _format_user_message(query: str, specimen_id: int | None) -> str:
    if specimen_id is None:
        return f"Query: {query}\n\nProceed with FM tool calls and a typed claim."
    return (
        f"Specimen id: {specimen_id}\n"
        f"Query: {query}\n\n"
        "Proceed with FM tool calls and a typed claim."
    )


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


def trajectories_to_sft_records(
    trajectories: Iterable[Trajectory],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    only_passing: bool = True,
) -> list[dict[str, Any]]:
    """Convert trajectories to SFT training records.

    Each record carries:
        - ``messages``: full chat history including assistant turns.
        - ``run_id``, ``specimen_id``: provenance.
        - ``aggregate_decision``: the final verifier decision.
        - ``passing``: True iff verifier returned PASS.
    """
    out: list[dict[str, Any]] = []
    for t in trajectories:
        passing = (
            t.final_verdict is not None
            and t.final_verdict.aggregate_decision is SourceDecision.PASS
        )
        if only_passing and not passing:
            continue
        out.append({
            "run_id": t.run_id,
            "specimen_id": t.specimen_id,
            "aggregate_decision": (
                t.final_verdict.aggregate_decision.value
                if t.final_verdict is not None else None
            ),
            "passing": passing,
            "messages": trajectory_to_messages(t, system_prompt=system_prompt),
        })
    return out


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def trajectories_to_dpo_pairs(
    trajectories: Iterable[Trajectory],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Build DPO preference pairs from passing and failing trajectories.

    Pairs share the same ``specimen_id`` and ``query``; ``chosen``
    comes from a passing trajectory, ``rejected`` from a failing one.
    Specimens that lack both kinds of trajectory get skipped.
    """
    by_specimen: dict[tuple[int | None, str], dict[str, list]] = {}
    for t in trajectories:
        key = (t.specimen_id, t.query)
        bucket = by_specimen.setdefault(key, {"pass": [], "fail": []})
        if (
            t.final_verdict is not None
            and t.final_verdict.aggregate_decision is SourceDecision.PASS
        ):
            bucket["pass"].append(t)
        elif (
            t.final_verdict is not None
            and t.final_verdict.aggregate_decision is SourceDecision.FAIL
        ):
            bucket["fail"].append(t)

    pairs: list[dict[str, Any]] = []
    for (sid, query), bucket in by_specimen.items():
        if not bucket["pass"] or not bucket["fail"]:
            continue
        for pos in bucket["pass"]:
            for neg in bucket["fail"]:
                prompt_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _format_user_message(query, sid)},
                ]
                # Concatenate just the assistant text from each trajectory.
                chosen = _concat_assistant_text(pos)
                rejected = _concat_assistant_text(neg)
                pairs.append({
                    "specimen_id": sid,
                    "query": query,
                    "prompt_messages": prompt_messages,
                    "chosen": chosen,
                    "rejected": rejected,
                })
    return pairs


def _concat_assistant_text(traj: Trajectory) -> str:
    """Glue together every assistant turn the LLM produced."""
    parts: list[str] = []
    for s in traj.steps:
        if s.llm_action is not None and s.llm_action.raw_text:
            parts.append(s.llm_action.raw_text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# GRPO prompt set
# ---------------------------------------------------------------------------


def trajectories_to_grpo_prompts(
    trajectories: Iterable[Trajectory],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """Build the prompt set GRPO will sample new completions for.

    Each record carries the initial system + user messages plus
    metadata (specimen_id, query). The trainer's reward function
    (in :mod:`fmllm.training.reward`) re-runs the verifier on each
    generated completion to compute the reward.
    """
    seen: set[tuple[int | None, str]] = set()
    out: list[dict[str, Any]] = []
    for t in trajectories:
        key = (t.specimen_id, t.query)
        if deduplicate and key in seen:
            continue
        seen.add(key)
        out.append({
            "specimen_id": t.specimen_id,
            "query": t.query,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _format_user_message(t.query, t.specimen_id)},
            ],
        })
    return out


__all__ = [
    "trajectories_to_dpo_pairs",
    "trajectories_to_grpo_prompts",
    "trajectories_to_sft_records",
    "trajectory_to_messages",
]
