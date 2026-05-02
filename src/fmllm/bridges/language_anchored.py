"""Language-anchored bridge: emits a natural-language caption per FM.

The captions paraphrase the same content the structure-preserving
bridge emits as a typed JSON object. The caption format is fixed per
FM so a complementary parser (:func:`parse_caption`) can recover the
numerical values, which round-trip tests and downstream consumers
depend on.

Caption shapes (one per FM):

    - FM1: ``"The image shows N atoms at positions (x1, y1), ... with
      confidence c1, ... within radius r LJ at 90% coverage."``
    - FM2: ``"The radial distribution function corresponds to a coarse-
      grained energy of E plus or minus sigma LJ units per atom."``
    - FM3: ``"The kinetic energy distribution has shape parameter alpha
      and scale parameter beta, consistent with a temperature of T
      plus or minus delta LJ units."``

Each caption ends with a one-line constraint summary that mirrors
the ``applicable_constraints`` block of the structure-preserving
output.

Depends on:
    torch.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from fmllm.bridges.base import (
    BaseBridge,
    FMContext,
    assemble_applicable_constraints,
)


def _to_scalar(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item())
    return float(x)


def _format_constraint_summary(applicable: list) -> str:
    if not applicable:
        return "Constraints: none recorded."
    parts = []
    for c in applicable:
        flag = "ok" if c.satisfied_in_training else "low"
        parts.append(f"{c.constraint_name}={c.satisfaction_score:.2f}/{flag}")
    return "Constraints: " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class LanguageAnchoredBridge(BaseBridge):
    """Abstract language-anchored bridge."""

    def emit(
        self,
        raw_output: dict[str, Any],
        *,
        input_provenance: dict[str, Any] | None = None,
        in_distribution: bool | None = None,
    ) -> str:
        body = self._build_caption(raw_output)
        applicable = assemble_applicable_constraints(self.context)
        constraint_line = _format_constraint_summary(applicable)
        provenance_line = ""
        if input_provenance:
            kv = ", ".join(f"{k}={v}" for k, v in input_provenance.items())
            provenance_line = f" (input: {kv})"
        flag = ""
        if in_distribution is False:
            flag = " The model flags this input as out-of-distribution."
        return f"{body}{provenance_line}{flag} {constraint_line}"

    # subclass hook
    def _build_caption(self, raw_output: dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FM1
# ---------------------------------------------------------------------------


class FM1LanguageBridge(LanguageAnchoredBridge):
    def __init__(self, context: FMContext, *, confidence_threshold: float = 0.5) -> None:
        super().__init__(context)
        self.confidence_threshold = confidence_threshold

    def _build_caption(self, raw_output: dict[str, Any]) -> str:
        count_logits = raw_output["count_logits"]
        positions = raw_output["positions"]
        conf_logits = raw_output["confidence_logits"]

        if isinstance(count_logits, torch.Tensor):
            count_logits = count_logits.detach().float().cpu()
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().float().cpu()
        if isinstance(conf_logits, torch.Tensor):
            conf_logits = conf_logits.detach().float().cpu()

        n_pred = int(torch.as_tensor(count_logits).argmax().item())
        confs = torch.sigmoid(torch.as_tensor(conf_logits))
        keep = confs > self.confidence_threshold
        kept_pos = torch.as_tensor(positions)[keep]
        kept_conf = confs[keep]

        if kept_pos.numel() == 0:
            pos_str = "no high-confidence query slots"
            conf_str = ""
        else:
            pos_strs = [
                f"({float(p[0]):.3f}, {float(p[1]):.3f})"
                for p in kept_pos
            ]
            conf_strs = [f"{float(c):.3f}" for c in kept_conf]
            pos_str = "positions " + ", ".join(pos_strs)
            conf_str = " with confidence " + ", ".join(conf_strs)

        radius_90 = self.context.calibration_threshold(0.10)
        radius_text = ""
        if radius_90 is not None:
            radius_text = f" within radius {float(radius_90):.3f} LJ at 90% coverage"

        return (
            f"The image shows {n_pred} atoms at "
            f"{pos_str}{conf_str}{radius_text}."
        )


# ---------------------------------------------------------------------------
# FM2
# ---------------------------------------------------------------------------


class FM2LanguageBridge(LanguageAnchoredBridge):
    def _build_caption(self, raw_output: dict[str, Any]) -> str:
        energy = _to_scalar(raw_output["energy"])
        q90 = self.context.calibration_threshold(0.10)
        if q90 is not None:
            return (
                f"The radial distribution function corresponds to a "
                f"coarse-grained energy of {energy:.4f} plus or minus "
                f"{float(q90):.4f} LJ units per atom (90% coverage)."
            )
        return (
            f"The radial distribution function corresponds to a "
            f"coarse-grained energy of {energy:.4f} LJ units per atom."
        )


# ---------------------------------------------------------------------------
# FM3
# ---------------------------------------------------------------------------


class FM3LanguageBridge(LanguageAnchoredBridge):
    def _build_caption(self, raw_output: dict[str, Any]) -> str:
        alpha = _to_scalar(raw_output["alpha"])
        beta = _to_scalar(raw_output["beta"])
        temp = alpha * beta
        q90 = self.context.calibration_threshold(0.10)
        # FM3's calibration is a per-specimen NLL threshold; we translate
        # the band on temperature heuristically by approximating the
        # asymptotic standard error of alpha * beta. For a Gamma fit on
        # n samples, sigma(mean) ~ beta * sqrt(alpha / n_eff). With
        # n_eff approximated from the calibration NLL gap, we get a
        # lightweight but useful caption.
        if q90 is not None and alpha > 0:
            # Approximate symmetric band width: scale sqrt(NLL margin).
            delta = float(beta) * float((q90 / max(alpha, 1.0e-6)) ** 0.5)
        else:
            delta = 0.0
        if delta > 0:
            return (
                f"The kinetic energy distribution has shape parameter "
                f"{alpha:.4f} and scale parameter {beta:.4f}, consistent "
                f"with a temperature of {temp:.4f} plus or minus "
                f"{delta:.4f} LJ units."
            )
        return (
            f"The kinetic energy distribution has shape parameter "
            f"{alpha:.4f} and scale parameter {beta:.4f}, consistent "
            f"with a temperature of {temp:.4f} LJ units."
        )


# ---------------------------------------------------------------------------
# Factory + parser
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[LanguageAnchoredBridge]] = {
    "fm1_image": FM1LanguageBridge,
    "fm2_rdf": FM2LanguageBridge,
    "fm3_traj": FM3LanguageBridge,
}


def make_language_bridge(
    context: FMContext, **kwargs: Any,
) -> LanguageAnchoredBridge:
    cls = _REGISTRY.get(context.fm_name)
    if cls is None:
        raise ValueError(
            f"no language-anchored bridge registered for fm_name={context.fm_name!r}; "
            f"known: {sorted(_REGISTRY)}"
        )
    return cls(context, **kwargs)


_FM1_HEAD_RE = re.compile(
    r"The image shows\s+(?P<n>\d+)\s+atoms"
)
_FM1_POS_RE = re.compile(
    r"\((?P<x>-?\d+\.\d+),\s*(?P<y>-?\d+\.\d+)\)"
)
_FM1_CONF_RE = re.compile(
    r"with confidence\s+(?P<confs>(?:\d+\.\d+(?:,\s*)?)+)"
)
_FM2_RE = re.compile(
    r"coarse-grained energy of\s+(?P<e>-?\d+\.\d+)"
)
_FM2_BAND_RE = re.compile(
    r"plus or minus\s+(?P<sigma>\d+\.\d+)\s+LJ units per atom"
)
_FM3_RE = re.compile(
    r"shape parameter\s+(?P<alpha>-?\d+\.\d+)\s+and scale parameter\s+(?P<beta>-?\d+\.\d+)"
)
_FM3_TEMP_RE = re.compile(
    r"temperature of\s+(?P<T>-?\d+\.\d+)(?:\s+plus or minus\s+(?P<dT>-?\d+\.\d+))?"
)


def parse_caption(caption: str, fm_name: str) -> dict[str, Any]:
    """Recover numerical values from a caption emitted by the language bridge.

    The function is tolerant to the constraint-summary tail and the
    optional provenance / out-of-distribution flag the base class
    appends. It extracts the FM-specific value payload only.
    """
    if fm_name == "fm1_image":
        head = _FM1_HEAD_RE.search(caption)
        if head is None:
            raise ValueError("FM1 caption: no atom-count head")
        n_pred = int(head.group("n"))
        positions = [
            (float(m.group("x")), float(m.group("y")))
            for m in _FM1_POS_RE.finditer(caption)
        ]
        conf_match = _FM1_CONF_RE.search(caption)
        confidences: list[float] = []
        if conf_match is not None:
            confidences = [
                float(x) for x in conf_match.group("confs").replace(",", " ").split()
                if x
            ]
        return {
            "n_atoms_pred": n_pred,
            "positions": positions,
            "confidences": confidences,
        }

    if fm_name == "fm2_rdf":
        m = _FM2_RE.search(caption)
        if m is None:
            raise ValueError("FM2 caption: no energy match")
        out: dict[str, Any] = {"energy_lj_per_atom": float(m.group("e"))}
        b = _FM2_BAND_RE.search(caption)
        if b is not None:
            out["band_sigma"] = float(b.group("sigma"))
        return out

    if fm_name == "fm3_traj":
        m = _FM3_RE.search(caption)
        if m is None:
            raise ValueError("FM3 caption: no shape/scale match")
        out = {"alpha": float(m.group("alpha")), "beta": float(m.group("beta"))}
        t = _FM3_TEMP_RE.search(caption)
        if t is not None:
            out["temperature_lj"] = float(t.group("T"))
            if t.group("dT") is not None:
                out["temperature_band"] = float(t.group("dT"))
        return out

    raise ValueError(f"unknown fm_name={fm_name!r}")


__all__ = [
    "FM1LanguageBridge",
    "FM2LanguageBridge",
    "FM3LanguageBridge",
    "LanguageAnchoredBridge",
    "make_language_bridge",
    "parse_caption",
]
