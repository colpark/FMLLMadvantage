"""Self-supervised FM2: masked-RDF reconstruction backbone.

Phase 10 follow-up to the Phase 9 negative result. The supervised
FM2 was trained end-to-end against per-atom potential energy, which
the probing study showed produced a representation with selective
richness (phase strongly encoded, atom count not). This subpackage
trains a parallel backbone on the same RDF inputs but with a
masked-bin reconstruction objective, on the hypothesis that a
representation shaped by predicting RDF structure (rather than one
scalar derived from it) carries more information the LLM can use.

The architecture is structurally identical to ``FM2RDFTransformer``
so that the existing connector code consumes it without changes:

    encode(rdf) -> (B, rdf_bins + 1, embed_dim)

The forward pass during pretraining accepts a mask tensor and
returns reconstructed bin values for the loss.

Entry points:
    FM2SSLTransformer    self-supervised model
    build_fm2_ssl_model  factory honoring the project config schema
"""

from fmllm.fms.fm2_rdf_ssl.model import FM2SSLTransformer, build_fm2_ssl_model

__all__ = ["FM2SSLTransformer", "build_fm2_ssl_model"]
