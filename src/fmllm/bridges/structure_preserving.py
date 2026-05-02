"""Structure-preserving bridge: emits :class:`BridgedFMOutput` per FM.

The bridge composes raw FM output with the static :class:`FMContext`
(metadata, probe report, calibration) into a typed Pydantic object
the verifier and the LLM consume through the same schema. The per-FM
extraction logic lives in dedicated subclasses; the abstract base
handles the shared boilerplate (assembling :class:`Source`,
:class:`ApplicableConstraint`, dependencies, timestamp).

The bridge emits:
    :class:`BridgedFMOutput` with:
        - ``prediction`` carrying a typed value payload
          (:class:`AtomSet`, :class:`EnergyPerAtom`, or
          :class:`GammaKEDistribution`),
        - ``source`` recording fm name / version / in-distribution flag,
        - ``applicable_constraints`` cross-referenced from the probe
          report and metadata,
        - ``dependencies`` materialized from the metadata's dependency
          edges with runtime-derived values where available,
        - ``timestamp`` in ISO 8601 UTC.

Use :func:`make_structure_bridge` to dispatch by FM name.

Depends on:
    torch, pydantic.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from fmllm.bridges.base import (
    BaseBridge,
    FMContext,
    assemble_applicable_constraints,
    assemble_dependencies,
)
from fmllm.fms._schemas import (
    BridgedFMOutput,
    Prediction,
    Source,
    Uncertainty,
)
from fmllm.fms._schemas.probe_schema import now_utc_iso
from fmllm.fms.fm1_image.bridge_schema import AtomPosition, AtomSet
from fmllm.fms.fm2_rdf.bridge_schema import EnergyPerAtom
from fmllm.fms.fm3_traj.bridge_schema import GammaKEDistribution


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class StructurePreservingBridge(BaseBridge):
    """Abstract structure-preserving bridge.

    Subclasses implement :meth:`_build_prediction` and
    :meth:`_derived_values`.
    """

    def emit(
        self,
        raw_output: dict[str, Any],
        *,
        input_provenance: dict[str, Any] | None = None,
        in_distribution: bool | None = None,
    ) -> BridgedFMOutput:
        prediction = self._build_prediction(raw_output)
        source = Source(
            fm_name=self.context.metadata.name,
            fm_version=self.context.metadata.version,
            in_distribution=bool(in_distribution) if in_distribution is not None else True,
            raw_input_provenance=input_provenance or {},
        )
        applicable = assemble_applicable_constraints(self.context)
        deps = assemble_dependencies(
            self.context,
            derived_values=self._derived_values(raw_output),
        )
        return BridgedFMOutput(
            prediction=prediction,
            source=source,
            applicable_constraints=applicable,
            dependencies=deps,
            timestamp=now_utc_iso(),
        )

    # ---- subclass hooks ---------------------------------------------------

    def _build_prediction(self, raw_output: dict[str, Any]) -> Prediction:  # pragma: no cover
        raise NotImplementedError

    def _derived_values(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# FM1: image -> AtomSet
# ---------------------------------------------------------------------------


def _to_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x)


class FM1StructureBridge(StructurePreservingBridge):
    """Bridge FM1 raw output (count_logits, positions, confidence_logits)
    into an :class:`AtomSet` payload."""

    def __init__(self, context: FMContext, *, confidence_threshold: float = 0.5) -> None:
        super().__init__(context)
        self.confidence_threshold = confidence_threshold

    def _build_atom_set(self, raw: dict[str, Any]) -> AtomSet:
        count_logits = _to_tensor(raw["count_logits"]).detach().float()
        positions = _to_tensor(raw["positions"]).detach().float()
        conf_logits = _to_tensor(raw["confidence_logits"]).detach().float()

        if count_logits.dim() != 1 or positions.dim() != 2 or conf_logits.dim() != 1:
            raise ValueError(
                "FM1 raw output must carry per-specimen tensors "
                "(count_logits: 1D, positions: (Q, 2), confidence_logits: 1D)"
            )

        confs = torch.sigmoid(conf_logits)
        keep = confs > self.confidence_threshold
        kept_positions = positions[keep]
        kept_confs = confs[keep]

        return AtomSet(
            n_atoms_pred=int(count_logits.argmax().item()),
            positions=[
                AtomPosition(
                    x_lj=float(p[0].item()),
                    y_lj=float(p[1].item()),
                    confidence=float(c.item()),
                )
                for p, c in zip(kept_positions, kept_confs, strict=True)
            ],
            raw_count_logits=[float(x.item()) for x in count_logits],
            raw_query_count=int(positions.shape[0]),
        )

    def _build_prediction(self, raw_output: dict[str, Any]) -> Prediction:
        atom_set = self._build_atom_set(raw_output)
        # Position uncertainty: a single calibrated radius applies to
        # every predicted atom.
        radius_90 = self.context.calibration_threshold(0.10)
        uncertainty: Uncertainty | None = None
        if radius_90 is not None:
            uncertainty = Uncertainty(
                lower=0.0,
                upper=float(radius_90),
                confidence_level=0.90,
            )
        return Prediction(
            quantity="atom_positions_lj",
            value=atom_set.model_dump(),
            units="lj_units",
            uncertainty=uncertainty,
        )

    def _derived_values(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        count_logits = _to_tensor(raw_output["count_logits"])
        return {
            "atom_count": int(count_logits.argmax().item()),
            "positions_in_box": True,  # the bridge doesn't enforce; verifier checks
        }


# ---------------------------------------------------------------------------
# FM2: RDF -> per-atom energy
# ---------------------------------------------------------------------------


class FM2StructureBridge(StructurePreservingBridge):
    """Bridge FM2 raw output (scalar energy) into an :class:`EnergyPerAtom`."""

    def _build_prediction(self, raw_output: dict[str, Any]) -> Prediction:
        energy = float(_to_tensor(raw_output["energy"]).detach().float().item())
        payload = EnergyPerAtom(value_lj=energy)
        q90 = self.context.calibration_threshold(0.10)
        uncertainty: Uncertainty | None = None
        if q90 is not None:
            uncertainty = Uncertainty(
                lower=energy - float(q90),
                upper=energy + float(q90),
                confidence_level=0.90,
            )
        return Prediction(
            quantity="per_atom_potential_energy",
            value=payload.model_dump(),
            units="lj_per_atom",
            uncertainty=uncertainty,
        )

    def _derived_values(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        # FM2 declares dependencies on atom_count (scales_with) and
        # temperature (derives). Without separate inputs the bridge
        # cannot fill these directly, so it leaves them as None and
        # the verifier consults FM1/FM3 for the matching values.
        return {}


# ---------------------------------------------------------------------------
# FM3: trajectory -> Gamma KE moments
# ---------------------------------------------------------------------------


class FM3StructureBridge(StructurePreservingBridge):
    """Bridge FM3 raw output (alpha, beta) into a
    :class:`GammaKEDistribution`."""

    def _build_prediction(self, raw_output: dict[str, Any]) -> Prediction:
        alpha = float(_to_tensor(raw_output["alpha"]).detach().float().item())
        beta = float(_to_tensor(raw_output["beta"]).detach().float().item())
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Gamma parameters must be positive, got alpha={alpha}, beta={beta}")
        mean = alpha * beta
        variance = alpha * beta * beta
        payload = GammaKEDistribution(
            alpha=alpha,
            beta=beta,
            mean=mean,
            variance=variance,
            implied_temperature_lj=mean,  # in 2D with unit mass, T = mean KE per atom
        )
        # Uncertainty for FM3 is not a numerical interval on (alpha, beta)
        # in general; the verifier checks the per-specimen empirical NLL
        # against the calibration threshold. The bridge surfaces the
        # threshold via context but does not put it in Uncertainty.
        return Prediction(
            quantity="kinetic_energy_distribution",
            value=payload.model_dump(),
            units="lj_per_atom",
            uncertainty=None,
        )

    def _derived_values(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        alpha = float(_to_tensor(raw_output["alpha"]).detach().float().item())
        beta = float(_to_tensor(raw_output["beta"]).detach().float().item())
        return {"temperature": alpha * beta}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[StructurePreservingBridge]] = {
    "fm1_image": FM1StructureBridge,
    "fm2_rdf": FM2StructureBridge,
    "fm3_traj": FM3StructureBridge,
}


def make_structure_bridge(
    context: FMContext, **kwargs: Any,
) -> StructurePreservingBridge:
    """Pick the right structure-preserving bridge subclass for a context."""
    cls = _REGISTRY.get(context.fm_name)
    if cls is None:
        raise ValueError(
            f"no structure-preserving bridge registered for fm_name={context.fm_name!r}; "
            f"known: {sorted(_REGISTRY)}"
        )
    return cls(context, **kwargs)


__all__ = [
    "FM1StructureBridge",
    "FM2StructureBridge",
    "FM3StructureBridge",
    "StructurePreservingBridge",
    "make_structure_bridge",
]
