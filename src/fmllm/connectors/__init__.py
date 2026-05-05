"""Phase 9 representation connectors.

A connector takes a frozen FM's hidden-state sequence and projects it
into the orchestrator LLM's input embedding space, so the LLM can
attend over the FM's representation directly. The connector is the
"vision encoder + projection" of LLaVA, ported to typed scientific
tools: the connector tokens flow into the LLM in addition to the
existing typed-JSON tool messages, not in place of them. The
verifier path is unchanged because the typed claim contract is
preserved.

Phase 9.A pilots a single connector on FM2 (RDF). FM1 and FM3 follow
if Phase 9.0 (probing) shows the energy-supervised representation
holds task-extra signal.

Submodules:
    qformer:           FM2Connector (Q-Former + projection)
    text_annotations:  templated per-specimen descriptions for Stage 1
                       alignment training
    dataset:           paired (rdf, text) Dataset

Depends on:
    torch, transformers (lazy via the training script).
"""

from fmllm.connectors.qformer import FM2Connector
from fmllm.connectors.text_annotations import (
    SpecimenAnnotation,
    annotate_specimen,
    annotate_specimen_from_h5,
)

__all__ = [
    "FM2Connector",
    "SpecimenAnnotation",
    "annotate_specimen",
    "annotate_specimen_from_h5",
]
