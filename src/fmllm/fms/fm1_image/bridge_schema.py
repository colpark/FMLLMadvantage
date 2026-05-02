"""Typed value payload for FM1 bridged outputs.

The structure-preserving bridge wraps an :class:`AtomSet` payload
inside a :class:`BridgedFMOutput.prediction.value` field. Bridges and
the verifier import :class:`AtomSet` to type-check FM1 outputs.

Produces:
    :class:`AtomSet` instances with predicted positions, confidence
    scores, and the matched query indices.

Depends on:
    pydantic.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field
from pydantic import BaseModel


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AtomPosition(_StrictModel):
    """One predicted atom in LJ units with its objectness score."""

    x_lj: float
    y_lj: float
    confidence: float = Field(ge=0.0, le=1.0)


class AtomSet(_StrictModel):
    """The set of predicted atoms with the count head's most likely value.

    The bridge populates ``positions`` after thresholding the
    confidence logits and matching to the count head's argmax.
    """

    n_atoms_pred: int = Field(ge=0)
    positions: list[AtomPosition] = Field(default_factory=list)
    raw_count_logits: list[float] = Field(default_factory=list)
    raw_query_count: int = Field(ge=0)


__all__ = ["AtomPosition", "AtomSet"]
