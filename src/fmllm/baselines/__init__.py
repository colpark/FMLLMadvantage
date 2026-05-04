"""Baseline systems for the Phase 8a comparison.

Three modes share the OHVD trajectory schema so the eight world-model
tests and the goal-accuracy metric can score them uniformly:

* **B0 (naked)**: a single LLM call with no FM tools and no verifier.
  The LLM gets a textual description of the dataset and is asked to
  commit ``n_atoms``, ``temperature``, ``motif`` from prior alone.
  See :func:`fmllm.baselines.naked.run_naked_baseline`.
* **B2 (no_verifier)**: the standard OHVD loop with a stub verifier
  that always returns ``PASS``. The LLM still calls FMs and sees the
  bridged outputs, but loses every signal that the multi-source
  verifier would normally surface.
  See :class:`fmllm.baselines.NoOpVerifier`.
* **B3 (full)**: the canonical Pipeline A from Phase 5; lives in
  ``fmllm.orchestrator``. The baseline runner just dispatches to it.

Naming follows the spec table from Phase 7's "where it should beat
baselines" discussion: B0 isolates whether grounding in FMs matters,
B2 isolates whether the verifier loop matters.
"""

from fmllm.baselines.naked import run_naked_baseline
from fmllm.baselines.noop_verifier import NoOpVerifier

__all__ = [
    "NoOpVerifier",
    "run_naked_baseline",
]
