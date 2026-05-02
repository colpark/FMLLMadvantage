"""Pydantic schema for the typed object the bridges emit.

The structure-preserving bridge transports raw FM output into a
:class:`BridgedFMOutput` that carries the prediction value, units,
calibrated uncertainty bounds, source metadata (FM name, version,
in-distribution flag), the constraint-satisfaction summary read off
the probe report, and the dependency edges declared in metadata. The
verifier and the LLM consume this object through the same schema, so
any change here ripples consistently downstream.

Produces:
    :class:`BridgedFMOutput` instances. JSON round-trip preserves
    every field.

Depends on:
    pydantic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictBridgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Uncertainty(_StrictBridgeModel):
    """Calibrated interval. ``lower`` and ``upper`` accept scalars or
    per-element lists for set-valued predictions."""

    lower: float | list[float]
    upper: float | list[float]
    confidence_level: float = Field(ge=0.0, le=1.0)


class Prediction(_StrictBridgeModel):
    """Predicted quantity with units and optional uncertainty.

    The ``value`` field stays generic so the per-FM ``bridge_schema.py``
    can carry typed payloads (atom set, scalar energy, Gamma moments).
    """

    quantity: str
    value: Any
    units: str
    uncertainty: Uncertainty | None = None


class Source(_StrictBridgeModel):
    """Provenance information for the prediction."""

    fm_name: str
    fm_version: str
    in_distribution: bool
    raw_input_provenance: dict[str, Any] = Field(default_factory=dict)


class ApplicableConstraint(_StrictBridgeModel):
    """One constraint relevant to this prediction with its probe score."""

    constraint_name: str
    type: str
    satisfied_in_training: bool
    satisfaction_score: float = Field(ge=0.0, le=1.0)


class BridgedDependency(_StrictBridgeModel):
    """One bridged dependency edge plus the value the FM derived."""

    target_variable: str
    relationship: str
    derived_value: Any
    confidence: float = Field(ge=0.0, le=1.0)


class BridgedFMOutput(_StrictBridgeModel):
    """Top-level typed output the bridges emit and the verifier reads."""

    prediction: Prediction
    source: Source
    applicable_constraints: list[ApplicableConstraint] = Field(default_factory=list)
    dependencies: list[BridgedDependency] = Field(default_factory=list)
    timestamp: str


__all__ = [
    "ApplicableConstraint",
    "BridgedDependency",
    "BridgedFMOutput",
    "Prediction",
    "Source",
    "Uncertainty",
]
