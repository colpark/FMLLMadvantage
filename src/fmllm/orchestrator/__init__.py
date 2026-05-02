"""LLM orchestration loop and trajectory data structures.

The Observe-Hypothesize-Verify-Decide loop drives a chat LLM through
repeated turns, dispatching FM tool calls into bridged outputs and
hypotheses through the multi-source verifier, until the model commits
a final claim or the step budget runs out.
"""

from fmllm.orchestrator.llm import (
    BaseLLM,
    MockLLM,
    TransformersLLM,
    parse_llm_response,
)
from fmllm.orchestrator.loop import DEFAULT_SYSTEM_PROMPT, FMRunnerFn, OHVDLoop
from fmllm.orchestrator.runners import (
    FM1Runner,
    FM2Runner,
    FM3Runner,
    FMRunner,
    build_runners_from_checkpoints,
)
from fmllm.orchestrator.trajectory import (
    ActionType,
    LLMAction,
    Step,
    StepType,
    TerminationReason,
    ToolCall,
    Trajectory,
)

__all__ = [
    "ActionType",
    "BaseLLM",
    "DEFAULT_SYSTEM_PROMPT",
    "FM1Runner",
    "FM2Runner",
    "FM3Runner",
    "FMRunner",
    "FMRunnerFn",
    "LLMAction",
    "MockLLM",
    "OHVDLoop",
    "Step",
    "StepType",
    "TerminationReason",
    "ToolCall",
    "Trajectory",
    "TransformersLLM",
    "build_runners_from_checkpoints",
    "parse_llm_response",
]
