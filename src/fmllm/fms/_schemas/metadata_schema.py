"""Pydantic schema for per-FM declarative metadata (Layer 1).

Each FM ships a ``metadata.yaml`` file that declares its modality,
input and output shapes, the physics constraints it should respect,
and the dependency edges between its prediction and other causal
variables in the system. The schema in this module validates those
files and exposes them to bridges, the verifier, and the evaluation
harness through a typed object.

Produces:
    Validated :class:`FMMetadata` instances loaded from disk.

Depends on:
    pydantic, pyyaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


ConstraintType = Literal["hard", "soft"]
RelationshipType = Literal["derives", "scales_with", "implies", "requires"]


class InputSchema(_StrictModel):
    """Description of the FM's input tensor.

    ``shape`` accepts integer dimensions and string placeholders such
    as ``"B"`` for the batch axis. ``normalization`` carries a free-form
    description that downstream code reads for documentation only.
    """

    shape: list[int | str]
    dtype: str = "float32"
    normalization: str = "raw"


class OutputSchema(_StrictModel):
    """Description of the FM's output."""

    output_type: str
    semantic_name: str
    units: str
    value_range: list[float] | None = None


class ConstraintDeclaration(_StrictModel):
    """One declared physics constraint plus its probe pointer."""

    name: str
    type: ConstraintType
    description: str
    expected_satisfaction: float = Field(ge=0.0, le=1.0)
    probe: str  # dotted module path: e.g. "fmllm.fms.fm1_image.probes.translation_equivariance"


class DependencyDeclaration(_StrictModel):
    """One dependency edge from this FM to another causal variable."""

    target_variable: str
    relationship: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""


class FMMetadata(_StrictModel):
    """Top-level FM metadata model."""

    name: str
    version: str
    modality: str
    input_schema: InputSchema
    output_schema: OutputSchema
    physics_constraints: list[ConstraintDeclaration] = Field(default_factory=list)
    dependencies: list[DependencyDeclaration] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


def load_fm_metadata(path: Path | str) -> FMMetadata:
    """Load and validate ``metadata.yaml`` at ``path``.

    Args:
        path: File path. The function rejects missing files immediately.

    Returns:
        The validated :class:`FMMetadata` instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        pydantic.ValidationError: If the YAML fails schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metadata YAML not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return FMMetadata.model_validate(raw)


__all__ = [
    "ConstraintDeclaration",
    "ConstraintType",
    "DependencyDeclaration",
    "FMMetadata",
    "InputSchema",
    "OutputSchema",
    "RelationshipType",
    "load_fm_metadata",
]
