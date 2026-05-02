"""Typed trajectory schemas for the OHVD loop.

A trajectory is an ordered list of typed steps the orchestrator
recorded while processing one query. Each step carries a
``step_type`` discriminator (observation, hypothesis,
verifier_verdict, final), a timestamp, and the relevant payload
(tool call, bridged FM output, claim, verifier verdict). The whole
trajectory serializes to JSON for downstream analysis and Phase 6
RL fine-tuning.

The :class:`LLMAction` type captures the LLM's parsed response per
turn, before the loop dispatches it. Phase 6's reward function reads
both the trajectory and the per-turn actions.

Depends on:
    pydantic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fmllm.fms._schemas import BridgedFMOutput
from fmllm.verifier.schema import PhysicalStateClaim, VerifierVerdict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StepType(str, Enum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    VERIFIER_VERDICT = "verifier_verdict"
    FINAL = "final"
    ERROR = "error"


class TerminationReason(str, Enum):
    COMMITTED = "committed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PARSE_FAILURE = "parse_failure"
    LLM_ERROR = "llm_error"


class ActionType(str, Enum):
    CALL_FM = "call_fm"
    HYPOTHESIZE = "hypothesize"
    COMMIT = "commit"
    ERROR = "error"


class ToolCall(_StrictModel):
    """The LLM's request to call one FM as a tool."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMAction(_StrictModel):
    """Parsed action from one LLM turn.

    Exactly one of ``tool_call``, ``claim`` (with `commit_final`), or
    ``error`` will be populated, matching ``action_type``.
    """

    action_type: ActionType
    tool_call: ToolCall | None = None
    claim: PhysicalStateClaim | None = None
    error: str | None = None
    raw_text: str = ""


class Step(_StrictModel):
    """One recorded step in the OHVD loop."""

    step_index: int
    step_type: StepType
    timestamp_utc: str
    llm_action: LLMAction | None = None
    bridged_output: BridgedFMOutput | None = None
    claim: PhysicalStateClaim | None = None
    verdict: VerifierVerdict | None = None
    notes: str = ""


class Trajectory(_StrictModel):
    """The full record of one orchestration run."""

    run_id: str
    query: str
    specimen_id: int | None = None
    started_utc: str
    finished_utc: str
    termination: TerminationReason
    final_claim: PhysicalStateClaim | None = None
    final_verdict: VerifierVerdict | None = None
    steps: list[Step] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def step_types(self) -> list[StepType]:
        return [s.step_type for s in self.steps]


__all__ = [
    "ActionType",
    "LLMAction",
    "Step",
    "StepType",
    "TerminationReason",
    "ToolCall",
    "Trajectory",
]
