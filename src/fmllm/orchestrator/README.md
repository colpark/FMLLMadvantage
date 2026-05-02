# fmllm.orchestrator

The LLM orchestration loop. Drives a chat LLM through the
Observe-Hypothesize-Verify-Decide cycle, dispatching FM tool calls
into bridged outputs and verifier verdicts back into the chat
context until the model commits a final claim or the step budget
exhausts.

## Files

- `trajectory.py` - typed step records and the top-level `Trajectory`
  Pydantic object the loop returns. Each step carries
  `step_type` (`observation`, `hypothesis`, `verifier_verdict`,
  `final`, `error`), the parsed `LLMAction`, and the relevant
  payload (bridged FM output, claim, verdict).
- `llm.py` - `BaseLLM` ABC, `MockLLM` for tests, `TransformersLLM`
  for remote (lazy-imports torch / transformers). `parse_llm_response`
  extracts the first JSON action from the model output and
  validates it against `LLMAction`.
- `loop.py` - `OHVDLoop` controller. Default system prompt
  describes the JSON action protocol. Calls the LLM up to
  `max_steps` times. Each iteration parses the LLM response,
  dispatches the action (call_fm / hypothesize / commit / error),
  records a step, and appends the tool result back into the chat.
- `runners.py` - `FM1Runner`, `FM2Runner`, `FM3Runner`. Each takes
  the model + bridge + dataset and exposes `runner(arguments) ->
  BridgedFMOutput`. `build_runners_from_checkpoints` loads all three
  models from the latest checkpoint under
  `checkpoints/<fm>/<scale>/<run_id>/` plus the matching bridge.

## Action protocol

The loop instructs the LLM to emit ONE JSON action per turn:

```
{"action": "call_fm", "tool_name": "fm1", "specimen_id": 42}
{"action": "hypothesize", "claim": {"n_atoms": 7, "temperature": 0.5}}
{"action": "commit", "claim": {"n_atoms": 7, "motif": "triangular_disk"}}
```

`parse_llm_response` is tolerant of leading or trailing prose. It
extracts the first JSON object in the response.

## Trajectory storage

Trajectories serialize to JSON via `Trajectory.model_dump_json()`.
The `scripts/run_pipeline.py` CLI saves them under
`runs/<run_id>/trajectory.json` alongside a manifest.

## Local vs remote

`MockLLM` makes the loop fully testable on a laptop without GPU or
LLM weights. `TransformersLLM` runs on the remote where Llama 3.1 8B
Instruct is available. The lazy import means importing
`fmllm.orchestrator` on a laptop does not pull in transformers.
