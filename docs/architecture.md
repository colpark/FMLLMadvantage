# Architecture

This document captures the high-level pipeline. Detail grows phase by
phase.

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

Every FM emits a typed Pydantic object that the bridges consume. Phase 3
defines the schema. The schema includes uncertainty bounds and the
physics constraints the FM respects.

### Bridge contract

Both bridges accept the typed FM output and emit an artifact the LLM
context-assembly logic can consume. The structure-preserving bridge
emits typed JSON with units and uncertainty. The language-anchored
bridge emits a natural-language caption. Phase 3 finalizes the
contract.

### Verifier verdict schema

The verifier's integrator emits a structured verdict with per-source
results, an aggregate decision, and a hint that names which sources
flagged issues. Phase 4 finalizes the schema.

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
