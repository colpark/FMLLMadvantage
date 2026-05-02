"""Shared Pydantic schemas for FM metadata, probe reports, and bridge outputs.

The three layers of constraint extraction (declarative metadata,
calibrated reliability, behavioral probes) all serialize through these
schemas. Downstream consumers (bridges, verifier, evaluation harness)
import the schema classes from here so the contract stays single-
sourced.
"""

from fmllm.fms._schemas.bridge_schema import (
    ApplicableConstraint,
    BridgedDependency,
    BridgedFMOutput,
    Prediction,
    Source,
    Uncertainty,
)
from fmllm.fms._schemas.metadata_schema import (
    ConstraintDeclaration,
    DependencyDeclaration,
    FMMetadata,
    InputSchema,
    OutputSchema,
    load_fm_metadata,
)
from fmllm.fms._schemas.probe_schema import (
    ProbeReport,
    ProbeResult,
    load_probe_report,
    save_probe_report,
)

__all__ = [
    "ApplicableConstraint",
    "BridgedDependency",
    "BridgedFMOutput",
    "ConstraintDeclaration",
    "DependencyDeclaration",
    "FMMetadata",
    "InputSchema",
    "OutputSchema",
    "Prediction",
    "ProbeReport",
    "ProbeResult",
    "Source",
    "Uncertainty",
    "load_fm_metadata",
    "load_probe_report",
    "save_probe_report",
]
