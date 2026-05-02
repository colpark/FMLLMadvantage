# Phase 5: Orchestration

## What I built

The orchestrator drives a chat LLM through the
Observe-Hypothesize-Verify-Decide cycle, dispatching FM tool calls
into bridged outputs and verifier verdicts back into the chat
context until the LLM commits or the step budget runs out.

### `fmllm.orchestrator`

- `trajectory.py` - typed step records:
  - `StepType` (`observation`, `hypothesis`, `verifier_verdict`,
    `final`, `error`).
  - `LLMAction` (parsed action with `action_type` ENUM).
  - `Step` and `Trajectory` Pydantic models. JSON round-trip preserves
    every field including nested bridged outputs and verdicts.
- `llm.py` -
  - `BaseLLM` ABC with one method: `chat(messages) -> str`.
  - `MockLLM` returns scripted responses in order; emits an error
    JSON when exhausted so the loop terminates cleanly.
  - `TransformersLLM` wraps Llama 3.1 8B Instruct (or any chat-
    template-compatible model) via lazy `transformers` import. The
    laptop never loads the heavy stack until an actual inference run
    starts.
  - `parse_llm_response(text)` extracts the first JSON object and
    validates it. Tolerates leading prose. Emits
    `LLMAction(action_type=ERROR)` on malformed input so the loop
    keeps going.
- `runners.py` - `FM1Runner`, `FM2Runner`, `FM3Runner`. Each takes
  `model + bridge + dataset` and exposes
  `runner(arguments) -> BridgedFMOutput`. The factory
  `build_runners_from_checkpoints` loads the latest run-id under
  `checkpoints/<fm>/<scale>/` for each FM and constructs the
  matching bridge via `load_fm_context`.
- `loop.py` - `OHVDLoop`:
  - System prompt explains the JSON action protocol (one of three
    shapes: `call_fm`, `hypothesize`, `commit`).
  - Each iteration: ask the LLM, parse the response, dispatch the
    action, record a step, append the result back into the chat.
  - `call_fm` -> runner -> bridged output -> tool message with
    a compact JSON summary the LLM can read.
  - `hypothesize` -> verifier -> verdict -> tool message with the
    aggregate decision and per-source breakdown.
  - `commit` -> verifier -> final verdict -> terminate.
  - `error` -> log + tool message prompting retry; the loop keeps
    going.
  - Honors `sources_config` per call so callers can pass
    `SourcesConfig.for_ablation("V0")` etc. for E4.
- `__init__.py` re-exports the public API.

### `scripts/run_pipeline.py`

End-to-end CLI for one specimen. Loads each FM model's checkpoint at
the requested training scale, builds bridges, opens the dataset,
constructs the verifier, instantiates the LLM (real or mock), runs
the loop, saves the trajectory + manifest under
`runs/<run_id>/`. Flags:

- `--specimen-id N` (required).
- `--query "..."` (default suffices for smoke tests).
- `--train-split train_50k` (matches checkpoint subtree).
- `--ablation V0..V4` (verifier sources_config preset).
- `--llm-model meta-llama/Llama-3.1-8B-Instruct` (or any chat model).
- `--mock-script scripts/mock_scripts/example.json` for a smoke test
  that needs no LLM weights.

### `scripts/mock_scripts/example.json`

Hard-coded 4-turn LLM response sequence the CLI consumes when
`--mock-script` is set. Calls FM1, FM2, FM3 then commits a 7-atom
triangular_disk claim. Useful for verifying remote wiring before
launching the real Llama inference path.

### Tests (`tests/test_orchestrator.py`, 18 tests, 160 total passing in 1.2s)

- `parse_llm_response`: 8 tests covering the three action shapes,
  leading prose, empty input, non-JSON input, unknown actions,
  malformed JSON.
- `MockLLM`: scripted responses pop in order; exhausted mock emits
  an error JSON.
- `OHVDLoop` end-to-end:
  - immediate commit terminates correctly.
  - sequential `call_fm` actions dispatch to the right runners and
    populate observation steps with the bridged outputs.
  - unknown tool name -> error step, loop continues.
  - parse error -> error step, loop continues.
  - LLM that never commits -> budget-exhausted termination.
  - verifier verdict appears in the LLM's chat context as a tool
    message with the expected JSON keys.
  - `sources_config` (`V0`) propagates through to the verifier and
    produces an aggregate-skip verdict.
  - full trajectory JSON round-trip preserves every field.

## What the user runs to verify Phase 5

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest -m "not gpu" -v
```

Expect 160 passing tests (142 from before + 18 new orchestrator tests).

### Remote 4xH100 host

#### Step 1. Mock-script smoke test (no LLM weights, GPU optional)

```
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_pipeline.py \
    --specimen-id 42 \
    --train-split train_50k \
    --mock-script scripts/mock_scripts/example.json \
    --ablation V4
```

Expect:
- A new `runs/<run_id>/` directory.
- `trajectory.json` with five steps (3 observations, 1 final, 1
  verifier verdict) and `termination = committed`.
- `manifest.yaml` recording paths and config.
- Wall clock ~30 seconds (FM forward passes are fast for one
  specimen each).

#### Step 2. Real Llama 3.1 8B inference

The first call downloads ~16 GB of weights (cached after).

```
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_pipeline.py \
    --specimen-id 42 \
    --train-split train_50k \
    --llm-model meta-llama/Llama-3.1-8B-Instruct \
    --ablation V4 \
    --step-budget 16
```

Expect:
- LLM weights load on first call (a few minutes).
- Per-turn inference latency ~1-3 seconds at default settings.
- Total wall clock per specimen ~30-60 seconds for a 16-step
  budget.
- A populated `trajectory.json` with assistant messages, observation
  steps, and a final verdict.

If the model is gated and the user has not authenticated, the load
fails with a friendly transformers error. Run
`huggingface-cli login` first.

## Known caveats

- **The mock script does not parameterize specimen_id.** The script
  bakes ``specimen_id: 42`` into every call_fm action. For arbitrary
  specimens, copy and edit the JSON.
- **The system prompt is conservative.** It instructs the LLM to
  emit one JSON action per turn and stay terse. Real Llama 3.1 may
  occasionally violate this; the parser tolerates leading prose and
  the loop logs parse errors as `error` steps without crashing.
- **`TransformersLLM` greedy / temperature=0.2 is the default.**
  Set `--llm-temperature 0.0` for fully deterministic runs.
- **Llama 3.1 8B is gated on Hugging Face.** Authenticate before
  the first run with `huggingface-cli login`. Alternatively, swap
  in any non-gated chat model via `--llm-model`.

## What remains for Phase 6 (RL fine-tuning)

- Implement `src/fmllm/training/`:
  - `trajectory_collection.py`: run Pipeline A across many
    specimens, filter to verifier-passing trajectories, write the
    RL training set.
  - `grpo_trainer.py`: GRPO fine-tune Llama 3.1 8B with LoRA.
  - `dpo_alternative.py`: DPO fallback.
- `scripts/train_pipeline_b.py` CLI.
- Tests on synthetic trajectories.
