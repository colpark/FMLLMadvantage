"""Pydantic types the verifier emits and consumes.

Top-level objects:
    - :class:`PhysicalStateClaim`: the LLM's typed claim about a
      specimen's underlying physical state. Phase 5's orchestrator
      populates this from a tool call.
    - :class:`SourcesConfig`: runtime activation map for the five
      verifier sources, with E4 ablation presets V0 to V4.
    - :class:`SourceVerdict`: one source's contribution to the verdict.
    - :class:`VerifierVerdict`: the integrator's aggregated result,
      including a structured :class:`Hint`.

Decisions are coarse-grained (`pass`, `fail`, `caveat`, `skip`) so the
LLM can react to them deterministically. The integrator combines
source decisions with a fixed precedence: any `fail` from a hard
source produces an aggregate `fail`; any `caveat` produces
`caveat`; otherwise `pass`.

Depends on:
    pydantic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceDecision(str, Enum):
    """Coarse outcome from one verifier source."""

    PASS = "pass"
    FAIL = "fail"
    CAVEAT = "caveat"
    SKIP = "skip"


class PhysicalStateClaim(_StrictModel):
    """The LLM's typed claim about a specimen's underlying state.

    Every field is optional. The LLM populates whichever components it
    is willing to commit to at this reasoning step. Sources skip the
    checks for fields the LLM left unset.
    """

    specimen_id: int | None = None
    n_atoms: int | None = None
    temperature: float | None = None
    motif: str | None = None
    positions: list[list[float]] | None = None
    per_atom_potential_energy: float | None = None
    notes: str = ""


class SourcesConfig(_StrictModel):
    """Which of the five verifier sources is active for a given call.

    The E4 ablation experiment varies this config across the same
    architecture, FMs, and bridges. The presets V0..V4 match the
    addendum's spec.
    """

    rule_library: bool = True
    literature: bool = True
    cross_fm: bool = True
    simulator: bool = True
    conformal: bool = True

    @classmethod
    def all(cls) -> SourcesConfig:
        return cls()

    @classmethod
    def none(cls) -> SourcesConfig:
        return cls(
            rule_library=False, literature=False, cross_fm=False,
            simulator=False, conformal=False,
        )

    @classmethod
    def for_ablation(cls, level: str) -> SourcesConfig:
        """E4 ablation presets.

        - ``V0``: no verification.
        - ``V1``: rules only.
        - ``V2``: rules + conformal.
        - ``V3``: rules + conformal + cross-FM.
        - ``V4``: all five sources.
        """
        if level == "V0":
            return cls.none()
        if level == "V1":
            return cls(
                rule_library=True, literature=False, cross_fm=False,
                simulator=False, conformal=False,
            )
        if level == "V2":
            return cls(
                rule_library=True, literature=False, cross_fm=False,
                simulator=False, conformal=True,
            )
        if level == "V3":
            return cls(
                rule_library=True, literature=False, cross_fm=True,
                simulator=False, conformal=True,
            )
        if level == "V4":
            return cls.all()
        raise ValueError(f"unknown ablation level {level!r}; use one of V0..V4")

    def active_sources(self) -> list[str]:
        out = []
        for name in ("rule_library", "literature", "cross_fm", "simulator", "conformal"):
            if getattr(self, name):
                out.append(name)
        return out


class SourceVerdict(_StrictModel):
    """One source's verdict on a claim and its bridged FM evidence."""

    source_name: str
    decision: SourceDecision
    confidence: float = Field(ge=0.0, le=1.0)
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class Hint(_StrictModel):
    """Structured guidance the LLM can act on after a non-pass verdict."""

    flagged_sources: list[str] = Field(default_factory=list)
    suggested_revisions: list[str] = Field(default_factory=list)
    direction: str = ""


class VerifierVerdict(_StrictModel):
    """Top-level verdict aggregated from per-source results."""

    aggregate_decision: SourceDecision
    source_verdicts: list[SourceVerdict] = Field(default_factory=list)
    hint: Hint = Field(default_factory=Hint)
    timestamp: str
    sources_config: SourcesConfig = Field(default_factory=SourcesConfig)


__all__ = [
    "Hint",
    "PhysicalStateClaim",
    "SourceDecision",
    "SourceVerdict",
    "SourcesConfig",
    "VerifierVerdict",
]
