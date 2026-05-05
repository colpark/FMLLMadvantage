"""Representation-level interpretability tools for FM2.

Phase 13 explores whether information that is *in* FM2's hidden state
but *not* exposed by the head or by the existing probe bank can be
surfaced to the orchestrator LLM as text. The pattern: train a sparse
autoencoder on FM2 CLS embeddings, label the resulting features by
their correlation with dataset attributes, and inject the most-active
labelled features per specimen into the LLM's prompt alongside the
existing PROBES payload.

Submodules:
    sae        TopKSAE module
    labels     correlation-based feature labelling

The architectural goal is the (input, representation, probe_outputs)
triangle: probes already cover the rep -> probe_outputs edge; this
module adds the rep -> text-label edge so the LLM can reason about
representation structure that probes don't pre-specify.
"""

from fmllm.representation.sae import TopKSAE, build_topk_sae

__all__ = ["TopKSAE", "build_topk_sae"]
