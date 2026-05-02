"""Bridge foundations: ``FMContext`` and ``BaseBridge``.

The bridges transport raw FM output into objects the LLM and the
verifier consume. Two flavors share one abstract base:

    - :class:`StructurePreservingBridge` emits a :class:`BridgedFMOutput`
      Pydantic object with typed values, units, calibrated uncertainty,
      and the constraint / dependency metadata declared in
      ``metadata.yaml``.
    - :class:`LanguageAnchoredBridge` emits a natural-language caption
      paraphrasing the same content.

Each bridge consumes an :class:`FMContext` that bundles the FM's
metadata, probe report, and conformal calibration. The context loads
once per checkpoint; bridges then wrap an unbounded number of forward-
pass results.

Produces:
    The :class:`BaseBridge` ABC plus helpers that the per-flavor
    modules subclass.

Depends on:
    pydantic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from fmllm.fms._schemas import (
    ApplicableConstraint,
    BridgedDependency,
    FMMetadata,
    ProbeReport,
)


@dataclass
class FMContext:
    """Per-FM bundle that bridges read at instantiation time.

    The context lives independent of any particular forward pass and
    carries the static information the bridge needs to decorate every
    prediction:

        - ``metadata``: declared constraints and dependencies.
        - ``probe_report``: empirical satisfaction scores from training.
        - ``calibration``: split-conformal thresholds keyed by alpha
          string (e.g. ``"0.1000"``). May be empty when the FM has not
          been calibrated yet.
    """

    fm_name: str
    metadata: FMMetadata
    probe_report: ProbeReport
    calibration: dict[str, Any]

    def calibration_threshold(self, alpha: float) -> float | None:
        """Look up the calibrated threshold at the given miscoverage level.

        Returns ``None`` if calibration is unavailable for this alpha.
        """
        thresholds = (self.calibration or {}).get("thresholds", {})
        # Stored as e.g. {"0.1000": 0.76}.
        return thresholds.get(f"{alpha:.4f}")

    def constraint_type(self, constraint_name: str) -> str | None:
        """Return the declared ``hard``/``soft`` type for a constraint."""
        for c in self.metadata.physics_constraints:
            if c.name == constraint_name:
                return c.type
        return None


class BaseBridge(ABC):
    """Abstract bridge over an :class:`FMContext`.

    Concrete subclasses implement :meth:`emit`, which transports a
    forward-pass dict into the bridge's output type.
    """

    def __init__(self, context: FMContext) -> None:
        self.context = context

    @property
    def fm_name(self) -> str:
        return self.context.fm_name

    @abstractmethod
    def emit(
        self,
        raw_output: dict[str, Any],
        *,
        input_provenance: dict[str, Any] | None = None,
        in_distribution: bool | None = None,
    ) -> Any:
        """Wrap a single FM forward-pass result.

        Args:
            raw_output: Dict of tensors / scalars the FM model returns.
            input_provenance: Optional metadata about the input
                (specimen ID, dataset path, etc.) the bridge records
                in the source field.
            in_distribution: Caller-supplied flag overriding any
                heuristic the bridge would otherwise apply. ``None``
                means the bridge keeps its default.

        Returns:
            A bridge-specific output: a ``BridgedFMOutput`` for the
            structure-preserving bridge, a string caption for the
            language-anchored bridge.
        """


def assemble_applicable_constraints(context: FMContext) -> list[ApplicableConstraint]:
    """Cross-reference probe-report scores with metadata constraint types.

    Both bridges call this helper, since the resulting list goes
    verbatim into the structure-preserving bridge and gets paraphrased
    by the language-anchored bridge.
    """
    out: list[ApplicableConstraint] = []
    for r in context.probe_report.results:
        ctype = context.constraint_type(r.constraint_name) or "soft"
        out.append(
            ApplicableConstraint(
                constraint_name=r.constraint_name,
                type=ctype,
                satisfied_in_training=bool(r.passes_threshold),
                satisfaction_score=float(r.satisfaction_score),
            )
        )
    return out


def assemble_dependencies(
    context: FMContext,
    derived_values: dict[str, Any] | None = None,
) -> list[BridgedDependency]:
    """Materialize the metadata's dependency edges for one prediction.

    ``derived_values`` carries per-target ``derived_value`` payloads the
    bridge computed at runtime (e.g. ``atom_count`` from FM1's count
    head). Targets without an entry default to ``None``.
    """
    derived_values = derived_values or {}
    out: list[BridgedDependency] = []
    for d in context.metadata.dependencies:
        out.append(
            BridgedDependency(
                target_variable=d.target_variable,
                relationship=d.relationship,
                derived_value=derived_values.get(d.target_variable),
                confidence=float(d.confidence),
            )
        )
    return out


__all__ = [
    "BaseBridge",
    "FMContext",
    "assemble_applicable_constraints",
    "assemble_dependencies",
]
