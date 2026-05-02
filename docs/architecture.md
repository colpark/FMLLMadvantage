# Architecture

This document captures the high-level pipeline. Detail grows phase by
phase.

The architectural commitment: constraints live with the foundation
models, not in an external knowledge base. Each FM ships its own
declared constraints (Layer 1 metadata), calibrated reliability bounds
(Layer 2 conformal calibration plus cross-FM tolerances), and
behavioral probes (Layer 3 probe report). The bridge layer composes
these into typed `BridgedFMOutput` objects. The verifier integrates
constraints from the FM bundles rather than supplying them. See
`docs/constraints.md` for the full pipeline plus per-FM examples.

The experimental program tests both the architectural claim (E1 to
E3) and the mechanistic claim (E4 verifier ablation, E5 FM quality
sweep). See `docs/experiments.md` for the five experiments with their
hypotheses, manipulations, measurements, and pass criteria.

## Pipeline at a glance

```
                 +---------+    +---------+    +---------+
                 | FM1 img |    | FM2 RDF |    | FM3 traj|
                 +----+----+    +----+----+    +----+----+
                      |              |              |
                      v              v              v
                +-----+-----+  +-----+-----+  +-----+-----+
                | bridges   |  | bridges   |  | bridges   |
                | (lang +   |  | (lang +   |  | (lang +   |
                |  struct)  |  |  struct)  |  |  struct)  |
                +-----+-----+  +-----+-----+  +-----+-----+
                      |              |              |
                      +--------+-----+--------+-----+
                               v              v
                         +-----+--------------+-----+
                         |   Llama orchestrator     |
                         |  (OHVD loop, Phase 5)    |
                         +-------------+------------+
                                       v
                         +-------------+------------+
                         |   Multi-source verifier  |
                         |  rule + lit + cross-fm + |
                         |  simulator + conformal   |
                         +-------------+------------+
                                       v
                                final claim
```

## Component contracts

The contracts live next to each component in its own README and module
docstrings. This document records only the cross-component contract
points.

### FM output schema

Every FM emits a typed value payload (see
`fmllm/fms/<fm>/bridge_schema.py`) plus a metadata-derived
`BridgedFMOutput` shell defined in
`fmllm/fms/_schemas/bridge_schema.py`. The shell carries the
prediction value, units, calibrated uncertainty, source provenance,
applicable constraints with probe scores, and dependency edges.

### Bridge contract

Both bridges accept the typed FM output and emit an artifact the LLM
context-assembly logic can consume. The structure-preserving bridge
emits a `BridgedFMOutput` JSON object. The language-anchored bridge
emits a natural-language caption that paraphrases the same content.
Phase 3 finalizes the implementation.

### Verifier verdict schema

The verifier's integrator emits a structured verdict with per-source
results, an aggregate decision, and a hint that names which sources
flagged issues. The integrator accepts a runtime `sources_config`
field so callers can disable individual sources for the E4 ablation.
Phase 4 finalizes the schema.

### Trajectory schema

The orchestrator records every step (observation, hypothesis, verdict,
final) in a typed structure that serializes to JSON for downstream
analysis and RL fine-tuning. Phase 5 finalizes the schema.

## Remote runtime topology

The remote host has 4 H100 80GB GPUs. The default placement is:
- GPU 0: FM1 training, then orchestration LLM at inference.
- GPU 1: FM2 training, then evaluation work.
- GPU 2: FM3 training.
- GPU 3: free for ad-hoc work and Pipeline B fine-tuning.

`CUDA_VISIBLE_DEVICES` overrides the placement at the script level.
