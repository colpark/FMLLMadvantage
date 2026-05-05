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
    causal     counterfactual interventions on SAE features (Phase 14)
    llm_sae    activation hooks + steerers for SAE on the LLM (Phase 15)

The architectural goal is the (input, representation, probe_outputs)
triangle: probes already cover the rep -> probe_outputs edge; this
module adds the rep -> text-label edge so the LLM can reason about
representation structure that probes don't pre-specify. Phase 14
adds the rep -> causal-effect edge so the labels we feed the LLM
are grounded in interventional rather than purely correlational
evidence. Phase 15 mirrors the recipe from FM2's representation onto
the LLM's residual stream itself, enabling the Templeton et al. /
Golden Gate Claude style of activation steering.
"""

from fmllm.representation.causal import (
    CausalEffect,
    Intervention,
    InterventionKind,
    audit_feature,
    filter_features_by_causal_effect,
)
from fmllm.representation.llm_sae import (
    ActivationHarvester,
    ActivationSteerer,
    resolve_layer_module,
)
from fmllm.representation.sae import TopKSAE, build_topk_sae

__all__ = [
    "ActivationHarvester",
    "ActivationSteerer",
    "CausalEffect",
    "Intervention",
    "InterventionKind",
    "TopKSAE",
    "audit_feature",
    "build_topk_sae",
    "filter_features_by_causal_effect",
    "resolve_layer_module",
]
