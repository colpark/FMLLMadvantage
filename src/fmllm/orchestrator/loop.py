"""The Observe-Hypothesize-Verify-Decide loop.

The loop drives an LLM through repeated calls until the model commits
a final claim or the step budget runs out. Each iteration:

    1. Build the chat messages and ask the LLM for one assistant turn.
    2. Parse the response into an :class:`LLMAction`.
    3. Dispatch the action:
         - ``call_fm``: invoke the registered runner, capture the
           bridged FM output, append a tool message with a JSON-
           serialized summary of the bridged output.
         - ``hypothesize``: hand the claim to the verifier, append
           a tool message with the verdict summary.
         - ``commit``: hand the claim to the verifier, record the
           final verdict, terminate.
         - ``error``: log a parse error and append a tool message
           prompting the LLM to try again.
    4. Append the appropriate :class:`Step` to the trajectory.

The loop exits when the LLM commits or when ``max_steps`` actions
have run, whichever comes first.

The conversation we hand the LLM holds:
    - one system message describing the task and the action protocol,
    - one user message echoing the query and any specimen-id
      context,
    - alternating assistant turns (one per LLM call) and tool turns
      (one per dispatched action), so the LLM sees its prior
      decisions and the resulting evidence.

Depends on:
    pydantic, fmllm.bridges, fmllm.verifier, fmllm.orchestrator.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from fmllm.fms._schemas import BridgedFMOutput
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
from fmllm.verifier import (
    PhysicalStateClaim,
    SourcesConfig,
    Verifier,
    VerifierVerdict,
)


FMRunnerFn = Callable[[dict[str, Any]], BridgedFMOutput]
"""Signature for an FM runner: arguments dict -> BridgedFMOutput."""


DEFAULT_SYSTEM_PROMPT = """\
You are a physics scientist analyzing a Lennard-Jones cluster specimen \
in a 2D synthetic testbed.

You can call three foundation models as tools:
  - fm1: image-based atom-set prediction (count + positions in LJ units).
  - fm2: radial-distribution-function-based per-atom potential energy.
  - fm3: trajectory-based Gamma kinetic-energy distribution.

After gathering FM evidence, you propose a typed claim about the \
underlying physical state. A claim has any subset of:
  - n_atoms (int between 5 and 30),
  - temperature (float in LJ units),
  - motif (one of "triangular_disk", "ring", "linear"),
  - per_atom_potential_energy (float).

Each turn, emit ONE JSON action and nothing else. Three shapes:

  {"action": "call_fm", "tool_name": "fm1", "specimen_id": <int>}
  {"action": "hypothesize", "claim": {"n_atoms": <int>, ...}}
  {"action": "commit", "claim": {"n_atoms": <int>, ...}}

The verifier evaluates each hypothesis and commits. If the verdict is \
fail or caveat, refine your claim. Commit only when confident.

You have a strict step budget. Use it wisely.
"""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bridged_summary(bridged: BridgedFMOutput) -> dict[str, Any]:
    """Compact JSON-friendly summary the LLM sees after a tool call."""
    pred = bridged.prediction.model_dump()
    return {
        "fm_name": bridged.source.fm_name,
        "in_distribution": bridged.source.in_distribution,
        "prediction": pred,
        "applicable_constraints": [
            {
                "name": ac.constraint_name,
                "type": ac.type,
                "score": ac.satisfaction_score,
                "satisfied": ac.satisfied_in_training,
            }
            for ac in bridged.applicable_constraints
        ],
        "dependencies": [
            d.model_dump() for d in bridged.dependencies
        ],
    }


def _verdict_summary(verdict: VerifierVerdict) -> dict[str, Any]:
    """Compact JSON-friendly summary of the verifier's verdict."""
    return {
        "aggregate_decision": verdict.aggregate_decision.value,
        "sources": [
            {
                "name": v.source_name,
                "decision": v.decision.value,
                "confidence": v.confidence,
                "message": v.message,
            }
            for v in verdict.source_verdicts
        ],
        "hint": verdict.hint.model_dump(),
    }


class OHVDLoop:
    """Observe-Hypothesize-Verify-Decide controller."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        verifier: Verifier,
        runners: dict[str, FMRunnerFn],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 16,
        sources_config: SourcesConfig | None = None,
    ) -> None:
        self.llm = llm
        self.verifier = verifier
        self.runners = runners
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.sources_config = sources_config

    def run(
        self,
        query: str,
        *,
        specimen_id: int | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Trajectory:
        run_id = run_id or generate_run_id("pipeline-a")
        started = _now_utc()

        bridged_outputs: list[BridgedFMOutput] = []
        steps: list[Step] = []
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._format_user_message(query, specimen_id),
            },
        ]

        termination = TerminationReason.BUDGET_EXHAUSTED
        final_claim: PhysicalStateClaim | None = None
        final_verdict: VerifierVerdict | None = None

        for step_idx in range(self.max_steps):
            try:
                raw_text = self.llm.chat(messages)
            except Exception as exc:  # noqa: BLE001
                steps.append(Step(
                    step_index=step_idx,
                    step_type=StepType.ERROR,
                    timestamp_utc=_now_utc(),
                    notes=f"LLM error: {exc!r}",
                ))
                termination = TerminationReason.LLM_ERROR
                break

            messages.append({"role": "assistant", "content": raw_text})
            action = parse_llm_response(raw_text)

            if action.action_type is ActionType.ERROR:
                steps.append(Step(
                    step_index=step_idx,
                    step_type=StepType.ERROR,
                    timestamp_utc=_now_utc(),
                    llm_action=action,
                    notes=action.error or "parse error",
                ))
                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "error": action.error or "parse error",
                        "hint": "respond with exactly one JSON action object",
                    }),
                })
                continue

            if action.action_type is ActionType.CALL_FM:
                step, bridged = self._dispatch_call_fm(action, step_idx)
                steps.append(step)
                if bridged is not None:
                    bridged_outputs.append(bridged)
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(_bridged_summary(bridged)),
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": step.notes}),
                    })
                continue

            if action.action_type in (ActionType.HYPOTHESIZE, ActionType.COMMIT):
                claim = action.claim
                assert claim is not None
                verdict = self.verifier.verify(
                    bridged_outputs, claim,
                    sources_config=self.sources_config,
                )
                steps.append(Step(
                    step_index=step_idx,
                    step_type=(
                        StepType.FINAL
                        if action.action_type is ActionType.COMMIT
                        else StepType.HYPOTHESIS
                    ),
                    timestamp_utc=_now_utc(),
                    llm_action=action,
                    claim=claim,
                ))
                steps.append(Step(
                    step_index=step_idx,
                    step_type=StepType.VERIFIER_VERDICT,
                    timestamp_utc=_now_utc(),
                    verdict=verdict,
                ))
                messages.append({
                    "role": "tool",
                    "content": json.dumps(_verdict_summary(verdict)),
                })
                if action.action_type is ActionType.COMMIT:
                    final_claim = claim
                    final_verdict = verdict
                    termination = TerminationReason.COMMITTED
                    break
                continue
        finished = _now_utc()

        return Trajectory(
            run_id=run_id,
            query=query,
            specimen_id=specimen_id,
            started_utc=started,
            finished_utc=finished,
            termination=termination,
            final_claim=final_claim,
            final_verdict=final_verdict,
            steps=steps,
            metadata=metadata or {},
        )

    # ----- internals -------------------------------------------------------

    def _format_user_message(
        self, query: str, specimen_id: int | None,
    ) -> str:
        if specimen_id is None:
            return f"Query: {query}\n\nProceed with FM tool calls and a typed claim."
        return (
            f"Specimen id: {specimen_id}\n"
            f"Query: {query}\n\n"
            "Proceed with FM tool calls and a typed claim."
        )

    def _dispatch_call_fm(
        self, action: LLMAction, step_idx: int,
    ) -> tuple[Step, BridgedFMOutput | None]:
        assert action.tool_call is not None
        tool_name = action.tool_call.tool_name
        runner = self.runners.get(tool_name)
        if runner is None:
            return (
                Step(
                    step_index=step_idx,
                    step_type=StepType.ERROR,
                    timestamp_utc=_now_utc(),
                    llm_action=action,
                    notes=f"unknown tool {tool_name!r}",
                ),
                None,
            )
        try:
            bridged = runner(action.tool_call.arguments)
        except Exception as exc:  # noqa: BLE001
            return (
                Step(
                    step_index=step_idx,
                    step_type=StepType.ERROR,
                    timestamp_utc=_now_utc(),
                    llm_action=action,
                    notes=f"runner {tool_name} raised {exc!r}",
                ),
                None,
            )
        return (
            Step(
                step_index=step_idx,
                step_type=StepType.OBSERVATION,
                timestamp_utc=_now_utc(),
                llm_action=action,
                bridged_output=bridged,
            ),
            bridged,
        )


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FMRunnerFn",
    "OHVDLoop",
]
