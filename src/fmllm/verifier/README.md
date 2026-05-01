# fmllm.verifier

The multi-source verifier.

Phase 4 will add:
- `sources/rule_library.py` - hard-coded rule checks on typed outputs.
- `sources/literature.py` - lookup against a curated database of
  canonical LJ cluster structures.
- `sources/cross_fm.py` - pairwise consistency checks between FM
  outputs.
- `sources/simulator.py` - forward MD rollout from the LLM's proposed
  state.
- `sources/conformal.py` - per-FM conformal prediction-band checks.
- `integrator.py` - the main verifier class that aggregates the five
  sources into a structured verdict with hints.

Currently empty.
