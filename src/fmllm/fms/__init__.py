"""Foundation models: FM1 (image), FM2 (RDF), FM3 (trajectory).

Each FM lives in its own sub-subpackage with model code, training
script, and conformal calibration. Common utilities live in
``fms.common``.
"""

from fmllm.fms.fm1_image import FM1ImageViT, build_fm1_model
from fmllm.fms.fm2_rdf import FM2RDFTransformer, build_fm2_model
from fmllm.fms.fm3_traj import FM3TrajTransformer, build_fm3_model

__all__ = [
    "FM1ImageViT",
    "FM2RDFTransformer",
    "FM3TrajTransformer",
    "build_fm1_model",
    "build_fm2_model",
    "build_fm3_model",
]
