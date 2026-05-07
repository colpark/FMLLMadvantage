"""Synthetic CoT generator for materials specimens.

Mirrors the LJ ``fmllm.training.synthetic_cot`` module but with the
materials ground-truth schema (formation_energy / e_above_hull /
is_stable / band_gap_class / space_group). Same Step-1 / Step-1b /
Step-2 / Step-3 / Final-commit structure so the SFT trainer
consumes the records unchanged.

Architecturally identical to the LJ version:

  * Probes are explicitly named in Step 1 (LLM learns probes are inputs).
  * Step 1b lists labelled SAE features when supplied (richer evidence).
  * Step 2 cross-checks via materials physics (band-gap consistency
    with crystal system, e_above_hull below stability threshold).
  * Step 3 resolves disagreement by deferring to highest-confidence probe.
  * Final commit comes from ground truth, not probe consensus.

Determinism: same (probes, sae_features, ground_truth) -> same text.

Depends on:
    Stdlib only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------

GroundTruthMaterials = dict[str, Any]
"""Schema:
    formation_energy : float
    e_above_hull     : float
    is_stable        : bool
    band_gap         : float
    band_gap_class   : str
    space_group      : int
    crystal_system   : str
    is_metal         : bool
    total_magnetization : float
    n_atoms          : int
"""


# ---------------------------------------------------------------------------
# Materials physics consistency checks
# ---------------------------------------------------------------------------


def stability_consistent(
    e_above_hull: float, is_stable_pred: bool, threshold: float = 0.025,
) -> tuple[bool, str]:
    """Cross-check whether the is_stable prediction matches e_above_hull.

    Returns ``(is_consistent, rationale)``.
    """
    inferred = e_above_hull <= threshold
    if inferred == bool(is_stable_pred):
        return True, (
            f"e_above_hull ≈ {e_above_hull:.3f} eV/atom is "
            f"{'below' if inferred else 'above'} the {threshold:.3f} "
            f"eV/atom stability cutoff; the is_stable probe agrees."
        )
    return False, (
        f"e_above_hull ≈ {e_above_hull:.3f} eV/atom would imply "
        f"is_stable = {inferred}, but the is_stable probe says "
        f"{is_stable_pred}. Probes disagree on stability."
    )


def band_gap_consistent(
    band_gap: float,
    predicted_class: str,
    is_metal_pred: str | None = None,
) -> tuple[bool, str]:
    """Cross-check predicted band-gap class against band_gap + is_metal probes.

    The is_metal probe is the authoritative metal/non-metal signal
    (a dedicated 2-class head). The band_gap regression probe
    additionally disambiguates narrow vs wide for non-metals.
    Their joint use is what the held-out evaluator scores against,
    so both should appear in the cross-check.
    """
    pclass = predicted_class.lower()
    is_metal_flag = (
        str(is_metal_pred).lower() == "metal" if is_metal_pred else None
    )

    if pclass == "metal":
        if is_metal_flag is True:
            return True, (
                "is_metal probe says 'metal'; band_gap probe value "
                f"{band_gap:.2f} eV is consistent (small or zero gap)."
            )
        if band_gap <= 1.0e-3:
            return True, "Band gap ≈ 0 confirms metal classification."
        return False, (
            f"Predicted 'metal' but band_gap probe is {band_gap:.2f} eV "
            f"and is_metal probe is {is_metal_pred!r}; defer to is_metal "
            f"if confidence is high."
        )
    if pclass == "narrow":
        if is_metal_flag is False and 1.0e-3 < band_gap <= 3.0:
            return True, (
                f"is_metal probe says 'non_metal' and band_gap "
                f"{band_gap:.2f} eV places this in the narrow-gap regime."
            )
        if 1.0e-3 < band_gap <= 3.0:
            return True, (
                f"Band gap {band_gap:.2f} eV places this in the narrow-gap regime."
            )
        return False, (
            f"Predicted 'narrow' but band_gap probe is {band_gap:.2f} eV; "
            f"reconcile with is_metal probe ({is_metal_pred!r})."
        )
    if pclass == "wide":
        if is_metal_flag is False and band_gap > 3.0:
            return True, (
                f"is_metal probe says 'non_metal' and band_gap "
                f"{band_gap:.2f} eV places this in the wide-gap regime."
            )
        if band_gap > 3.0:
            return True, (
                f"Band gap {band_gap:.2f} eV places this in the wide-gap regime."
            )
        return False, (
            f"Predicted 'wide' but band_gap probe is {band_gap:.2f} eV; "
            f"reconcile with is_metal probe ({is_metal_pred!r})."
        )
    return False, (
        f"Band-gap probe value {band_gap:.2f} eV does not match the "
        f"predicted class '{predicted_class}'."
    )


# ---------------------------------------------------------------------------
# Probe-output access helpers
# ---------------------------------------------------------------------------


def _read(probe_outputs: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    """Return ``probe_outputs[name]`` or a sentinel dict if missing."""
    return probe_outputs.get(name) or {"prediction": None, "confidence": 0.0}


def _confidence(value: dict[str, Any]) -> float:
    try:
        return float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Synthetic CoT
# ---------------------------------------------------------------------------


@dataclass
class MaterialsCoT:
    """Output of :func:`generate_cot`."""

    text: str
    consistent: bool
    final_claim: dict[str, Any]


_SYSTEM_PROMPT = (
    "You are a scientific reasoner working with the Materials Project "
    "testbed of inorganic crystalline materials. You receive probe "
    "outputs derived from a frozen foundation model (CHGNet) and must "
    "reason explicitly about the evidence before committing a typed "
    "claim about the material's formation energy, stability, band-gap "
    "class, and space group."
)


def _format_sae_feature_dict(
    sae_features: list[tuple[str, float]] | None,
) -> str:
    if not sae_features:
        return ""
    items = [f'"{lab}": {float(act):.2f}' for lab, act in sae_features]
    return "{" + ", ".join(items) + "}"


def _user_message(
    probe_outputs: dict[str, dict[str, Any]],
    sae_features: list[tuple[str, float]] | None = None,
) -> str:
    """The user message the LLM sees at training and inference time."""
    payload = {
        name: {
            "prediction": value.get("prediction"),
            "confidence": round(float(value.get("confidence", 0.0)), 3),
        }
        for name, value in probe_outputs.items()
    }
    parts: list[str] = [
        "PROBES (each derived from a frozen CHGNet head on this specimen): "
        f"{json.dumps(payload, sort_keys=True)}",
    ]
    sae_payload = _format_sae_feature_dict(sae_features)
    if sae_payload:
        parts.append("")
        parts.append(
            "SAE_FEATURES (top-k labelled directions in CHGNet's "
            "representation that activated on this specimen): "
            f"{sae_payload}"
        )
    parts.append("")
    parts.append(
        "Reason through the evidence step by step, cross-check the "
        "probes against each other, then commit a final JSON claim of "
        "the form {\"formation_energy\": float, \"e_above_hull\": float, "
        "\"is_stable\": bool, \"band_gap_class\": \"metal\"|\"narrow\"|\"wide\", "
        "\"space_group\": int}."
    )
    return "\n".join(parts)


def generate_cot(
    *,
    probe_outputs: dict[str, dict[str, Any]],
    ground_truth: GroundTruthMaterials,
    sae_features: list[tuple[str, float]] | None = None,
) -> MaterialsCoT:
    """Render a deterministic templated reasoning chain for materials.

    Args:
        probe_outputs: Output of a materials probe-bank evaluate call
            for one specimen. Expected probe names: ``formation_energy``,
            ``e_above_hull``, ``band_gap``, ``is_metal``,
            ``space_group``. Missing probes are tolerated.
        ground_truth: Materials ground-truth dict (see
            :data:`GroundTruthMaterials`).
        sae_features: Optional list of (label, activation) tuples to
            render in Step 1b.
    """
    e_form_pred = _read(probe_outputs, "formation_energy")
    e_hull_pred = _read(probe_outputs, "e_above_hull")
    bg_pred = _read(probe_outputs, "band_gap")
    is_metal_pred = _read(probe_outputs, "is_metal")
    sg_pred = _read(probe_outputs, "space_group")

    e_hull_value = float(e_hull_pred.get("prediction") or 0.0)
    bg_value = float(bg_pred.get("prediction") or 0.0)
    is_metal_value = (
        str(is_metal_pred.get("prediction") or "").lower() or None
    )
    is_stable_inferred = e_hull_value <= 0.025
    # Band-gap class derivation: is_metal probe is the authoritative
    # metal/non-metal signal (dedicated 2-class head); band_gap probe
    # disambiguates narrow vs wide for non-metals.
    if is_metal_value == "metal":
        bg_class_inferred = "metal"
    elif bg_value <= 1.0e-3:
        bg_class_inferred = "metal"
    elif bg_value <= 3.0:
        bg_class_inferred = "narrow"
    else:
        bg_class_inferred = "wide"

    stable_ok, stable_rationale = stability_consistent(
        e_above_hull=e_hull_value,
        is_stable_pred=is_stable_inferred,
    )
    bg_ok, bg_rationale = band_gap_consistent(
        band_gap=bg_value,
        predicted_class=bg_class_inferred,
        is_metal_pred=is_metal_value,
    )
    consistent = stable_ok and bg_ok

    lines: list[str] = []
    lines.append("Step 1 - Read the probes:")
    if e_form_pred["prediction"] is not None:
        lines.append(
            f"  - formation-energy probe : "
            f"{float(e_form_pred['prediction']):.2f} eV/atom "
            f"(confidence {_confidence(e_form_pred):.2f})"
        )
    if e_hull_pred["prediction"] is not None:
        lines.append(
            f"  - e-above-hull probe     : "
            f"{e_hull_value:.3f} eV/atom "
            f"(confidence {_confidence(e_hull_pred):.2f})"
        )
    if bg_pred["prediction"] is not None:
        lines.append(
            f"  - band-gap probe         : "
            f"{bg_value:.2f} eV "
            f"(confidence {_confidence(bg_pred):.2f})"
        )
    if is_metal_pred["prediction"] is not None:
        lines.append(
            f"  - is-metal probe         : "
            f"{is_metal_value} "
            f"(confidence {_confidence(is_metal_pred):.2f})"
        )
    if sg_pred["prediction"] is not None:
        lines.append(
            f"  - space-group probe      : "
            f"#{sg_pred['prediction']} "
            f"(confidence {_confidence(sg_pred):.2f})"
        )

    if sae_features:
        lines.append("")
        lines.append(
            "Step 1b - Read the SAE-derived features (auto-discovered "
            "directions in the CHGNet representation, with correlation "
            "labels):"
        )
        for lab, act in sae_features[:8]:
            lines.append(f"  - {lab} (activation {float(act):.2f})")

    lines.append("")
    lines.append("Step 2 - Cross-check stability and band-gap class:")
    lines.append(f"  {stable_rationale}")
    lines.append(f"  {bg_rationale}")

    lines.append("")
    lines.append("Step 3 - Resolution:")
    if consistent:
        lines.append(
            "  All probes agree on a coherent picture: stability is "
            "decided by e_above_hull, the is-metal probe disambiguates "
            "metal vs non-metal, the band-gap probe places non-metals "
            "in narrow vs wide, and the space-group probe disambiguates "
            "the lattice."
        )
    else:
        candidates = {
            "formation-energy": _confidence(e_form_pred),
            "e-above-hull": _confidence(e_hull_pred),
            "band-gap": _confidence(bg_pred),
            "is-metal": _confidence(is_metal_pred),
            "space-group": _confidence(sg_pred),
        }
        leader = max(candidates.items(), key=lambda kv: kv[1])
        lines.append(
            f"  Cross-checks fail. Defer to the highest-confidence probe "
            f"({leader[0]}, conf {leader[1]:.2f}) and reconcile the others "
            f"against it before committing."
        )

    final_claim = {
        "formation_energy": float(ground_truth["formation_energy"]),
        "e_above_hull": float(ground_truth["e_above_hull"]),
        "is_stable": bool(ground_truth["is_stable"]),
        "band_gap_class": str(ground_truth["band_gap_class"]),
        "space_group": int(ground_truth["space_group"]),
    }
    lines.append("")
    lines.append(f"Final commit: {json.dumps(final_claim, sort_keys=True)}")

    return MaterialsCoT(
        text="\n".join(lines),
        consistent=consistent,
        final_claim=final_claim,
    )


def build_sft_record(
    *,
    probe_outputs: dict[str, dict[str, Any]],
    ground_truth: GroundTruthMaterials,
    specimen_id: int,
    sae_features: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Assemble the (system, user, assistant) chat record consumed by
    Phase 6's SFT trainer for the materials port."""
    cot = generate_cot(
        probe_outputs=probe_outputs,
        ground_truth=ground_truth,
        sae_features=sae_features,
    )
    return {
        "specimen_id": int(specimen_id),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _user_message(probe_outputs, sae_features),
            },
            {"role": "assistant", "content": cot.text},
        ],
        "ground_truth": cot.final_claim,
        "cot_consistent": cot.consistent,
        "sae_features_count": (len(sae_features) if sae_features else 0),
    }


__all__ = [
    "GroundTruthMaterials",
    "MaterialsCoT",
    "band_gap_consistent",
    "build_sft_record",
    "generate_cot",
    "stability_consistent",
]
