# fmllm.orchestrator

LLM orchestration loop.

Phase 5 will add:
- `llm.py` - thin wrapper around the local LLM (default Llama 3.1 8B
  Instruct loaded via vLLM or transformers).
- `loop.py` - the Observe-Hypothesize-Verify-Decide loop driven by a
  step budget and a verifier.
- `trajectory.py` - typed step structures that serialize trajectories
  to JSON for downstream analysis.

Currently empty.
