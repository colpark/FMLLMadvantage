"""Bridges that transport raw FM output into LLM-consumable artifacts.

Two flavors share an abstract base. The structure-preserving bridge
emits a typed Pydantic :class:`BridgedFMOutput` per prediction. The
language-anchored bridge emits a natural-language caption paraphrasing
the same content. Both consume an :class:`FMContext` that loads
metadata, probe report, and conformal calibration once per checkpoint.
"""

from fmllm.bridges.base import (
    BaseBridge,
    FMContext,
    assemble_applicable_constraints,
    assemble_dependencies,
)
from fmllm.bridges.compose import load_fm_context, metadata_yaml_path
from fmllm.bridges.language_anchored import (
    FM1LanguageBridge,
    FM2LanguageBridge,
    FM3LanguageBridge,
    LanguageAnchoredBridge,
    make_language_bridge,
    parse_caption,
)
from fmllm.bridges.structure_preserving import (
    FM1StructureBridge,
    FM2StructureBridge,
    FM3StructureBridge,
    StructurePreservingBridge,
    make_structure_bridge,
)

__all__ = [
    "BaseBridge",
    "FM1LanguageBridge",
    "FM1StructureBridge",
    "FM2LanguageBridge",
    "FM2StructureBridge",
    "FM3LanguageBridge",
    "FM3StructureBridge",
    "FMContext",
    "LanguageAnchoredBridge",
    "StructurePreservingBridge",
    "assemble_applicable_constraints",
    "assemble_dependencies",
    "load_fm_context",
    "make_language_bridge",
    "make_structure_bridge",
    "metadata_yaml_path",
    "parse_caption",
]
