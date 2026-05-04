"""Naked-LLM baseline (B0): one-shot commit, no FM tools, no verifier.

The B0 baseline isolates whether grounding in FMs matters at all. It
gives the LLM a textual description of the testbed (the same one the
full system uses) and asks it to commit ``n_atoms``, ``temperature``,
and ``motif`` from prior alone — no observations, no tool calls, no
verifier feedback.

The naked loop does exactly one LLM call per specimen, parses the
response, and writes a :class:`Trajectory` with one ``FINAL`` step.
The trajectory schema matches the OHVD output so the eight world-
model tests and the goal-accuracy metric score it without
modification.

Depends on:
    fmllm.orchestrator (LLM + Trajectory schema), fmllm.verifier.schema
    (claim type), fmllm.utils.run_ids.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fmllm.orchestrator.llm import BaseLLM, parse_llm_response
from fmllm.orchestrator.trajectory import (
    ActionType,
    LLMAction,
    Step,
    StepType,
    TerminationReason,
    Trajectory,
)
from fmllm.utils.run_ids import generate_run_id
from fmllm.verifier.schema import PhysicalStateClaim


NAKED_SYSTEM_PROMPT = """\
You are a physics scientist analyzing a Lennard-Jones cluster specimen \
in a 2D synthetic testbed.

You will NOT have access to any measurement tools or foundation models. \
You see only this prompt and a specimen identifier. From your prior \
knowledge of small 2D Lennard-Jones clusters in the testbed, you must \
commit to a single typed claim about the underlying physical state.

The dataset draws specimens from this distribution:
  - n_atoms: integer in [5, 25].
  - temperature: float in LJ units, roughly [0.1, 2.0].
  - motif: one of "triangular_disk", "ring", "linear".
  - per_atom_potential_energy: float in LJ units, typically in [-3, 0].

Emit exactly one JSON action with shape:

  {"action": "commit", "claim": {"n_atoms": <int>, "temperature": <float>, \
"motif": "<str>", "per_atom_potential_energy": <float>}}

No other text. No tool calls. One commit, then stop.
"""


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _format_user_message(query: str, specimen_id: int | None) -> str:
    parts = [query]
    if specimen_id is not None:
        parts.append(f"Specimen ID: {specimen_id}")
    parts.append("Commit your best guess as one JSON action.")
    return "\n\n".join(parts)


def run_naked_baseline(
    *,
    llm: BaseLLM,
    query: str,
    specimen_id: int,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    """Run the naked-LLM baseline on one specimen.

    The function makes a single LLM call. On a parseable commit it
    builds a one-step trajectory. On any other action type
    (``call_fm``, ``hypothesize``, ``error``) the trajectory records
    the parse failure and ``termination = PARSE_FAILURE``.
    """
    run_id = run_id or generate_run_id(f"baseline-naked-{specimen_id}")
    started = _now_utc()

    messages = [
        {"role": "system", "content": NAKED_SYSTEM_PROMPT},
        {"role": "user", "content": _format_user_message(query, specimen_id)},
    ]

    try:
        raw_text = llm.chat(messages)
    except Exception as exc:  # noqa: BLE001
        finished = _now_utc()
        return Trajectory(
            run_id=run_id,
            query=query,
            specimen_id=specimen_id,
            started_utc=started,
            finished_utc=finished,
            termination=TerminationReason.LLM_ERROR,
            steps=[
                Step(
                    step_index=0,
                    step_type=StepType.ERROR,
                    timestamp_utc=finished,
                    notes=f"LLM error: {exc!r}",
                )
            ],
            metadata={"baseline": "naked", **(metadata or {})},
        )

    action = parse_llm_response(raw_text)

    finished = _now_utc()
    base_trajectory = dict(
        run_id=run_id,
        query=query,
        specimen_id=specimen_id,
        started_utc=started,
        finished_utc=finished,
        metadata={"baseline": "naked", **(metadata or {})},
    )

    if action.action_type is ActionType.COMMIT and action.claim is not None:
        claim: PhysicalStateClaim = action.claim
        return Trajectory(
            **base_trajectory,
            termination=TerminationReason.COMMITTED,
            final_claim=claim,
            final_verdict=None,  # B0 has no verifier
            steps=[
                Step(
                    step_index=0,
                    step_type=StepType.FINAL,
                    timestamp_utc=finished,
                    llm_action=action,
                    claim=claim,
                )
            ],
        )

    # Anything else is a parse failure for B0's purposes.
    return Trajectory(
        **base_trajectory,
        termination=TerminationReason.PARSE_FAILURE,
        steps=[
            Step(
                step_index=0,
                step_type=StepType.ERROR,
                timestamp_utc=finished,
                llm_action=action,
                notes=(
                    action.error
                    if action.action_type is ActionType.ERROR
                    else f"unexpected action_type {action.action_type.value}"
                ),
            )
        ],
    )


__all__ = [
    "NAKED_SYSTEM_PROMPT",
    "run_naked_baseline",
]
