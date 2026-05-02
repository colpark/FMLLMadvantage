# scripts/mock_scripts/

Hard-coded LLM response sequences for local smoke testing of the
OHVD loop. Each file is a JSON list of strings; the orchestrator
treats each string as the LLM's response for one turn.

## Files

- `example.json` - 4-turn script: call FM1, FM2, FM3 in sequence,
  then commit a triangular_disk-7 claim.

## Use

```
uv run python scripts/run_pipeline.py \
    --specimen-id 42 \
    --mock-script scripts/mock_scripts/example.json
```

The mock CLI path requires no LLM weights and no GPU. Useful for
verifying the pipeline wiring before launching the real Llama 3.1
inference path on the remote.
