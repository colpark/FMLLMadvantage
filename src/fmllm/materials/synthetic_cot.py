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


# ---------------------------------------------------------------------------
# v2: rich CoT
# ---------------------------------------------------------------------------


def _format_rich_sae_features(
    sae_features: list[tuple[str, float]] | None,
    feature_metadata: dict | None,
) -> str:
    """Render Step 1b body for the rich CoT.

    feature_metadata: optional mapping str(idx) -> {
        "label_rich": "...",
        "tags": [...],
        "top_specimens": [{"formula": "...", ...}, ...],
        "activation_quantiles": {"p50": ..., "p90": ..., "p99": ..., "max": ...},
    }
    """
    if not sae_features:
        return ""
    lines: list[str] = []
    md = feature_metadata or {}
    for label, act in sae_features[:8]:
        # Try to extract feature index from labels like "f24: ..." or
        # "f24 (rare)" -- store under str key in feature_metadata.
        feat_idx_str = ""
        if isinstance(label, str) and label.startswith("f"):
            tail = label[1:]
            for i, c in enumerate(tail):
                if not c.isdigit():
                    break
                feat_idx_str = tail[: i + 1]
            else:
                feat_idx_str = tail
        meta = md.get(feat_idx_str, {}) if feat_idx_str else {}
        quantiles = meta.get("activation_quantiles") or {}
        top_specs = meta.get("top_specimens") or []
        rich_desc = meta.get("label_rich")

        # Calibrated-strength descriptor.
        strength = ""
        if quantiles:
            p99 = quantiles.get("p99")
            p90 = quantiles.get("p90")
            p50 = quantiles.get("p50")
            if p99 is not None and act >= p99:
                strength = " [99th+ pct: very strong firing]"
            elif p90 is not None and act >= p90:
                strength = " [90th+ pct: strong firing]"
            elif p50 is not None and act >= p50:
                strength = " [median: typical firing]"
            else:
                strength = " [below-median firing]"

        if rich_desc:
            lines.append(
                f"  - {rich_desc}; activation {float(act):.2f}{strength}"
            )
        elif top_specs:
            examples = ", ".join(
                str(s.get("formula") or s.get("material_id") or "?")
                for s in top_specs[:5]
            )
            lines.append(
                f"  - {label} (activation {float(act):.2f}{strength}); "
                f"fires on: {examples}"
            )
        else:
            lines.append(f"  - {label} (activation {float(act):.2f})")
    return "\n".join(lines)


def _compositional_sanity(ground_truth: GroundTruthMaterials) -> str:
    """Step 2 body: surface formula / n_atoms / crystal system as priors."""
    formula = ground_truth.get("formula") or "(unknown)"
    n_atoms = int(ground_truth.get("n_atoms", 0) or 0)
    crystal_system = ground_truth.get("crystal_system") or "(unknown)"
    space_group = int(ground_truth.get("space_group", -1) or -1)
    is_metal = bool(ground_truth.get("is_metal", False))
    bg_class = ground_truth.get("band_gap_class") or "(unknown)"

    chemistry_note = ""
    formula_lc = formula.lower()
    # A few cheap pattern hints for the templated prose. The LLM has its
    # own priors; these just nudge the reasoning to be specific about
    # composition rather than abstract.
    if any(elem in formula for elem in ("O", "S", "Se", "Te")) and any(
        m in formula for m in ("Ti", "Fe", "Co", "Ni", "Cu", "Zn", "Mn", "V")
    ):
        chemistry_note = (
            "  Composition contains a transition-metal cation paired "
            "with a chalcogenide anion; common motifs in this family "
            "include perovskites, spinels, and rocksalt-type oxides "
            "(many are wide- or narrow-gap semiconductors)."
        )
    elif formula in ("Si", "Ge", "C", "Sn"):
        chemistry_note = (
            "  Single-element formula in Group IV: the canonical "
            "ground state is the diamond-cubic structure (sg 227, "
            "Fd-3m) with a narrow indirect gap. Expect tetrahedral "
            "covalent bonding."
        )
    elif "H" in formula and len(formula) <= 5:
        chemistry_note = (
            "  Formula includes hydrogen in a small unit cell; common "
            "for hydrides and molecular crystals."
        )

    return (
        f"  Formula: {formula}. Unit cell: {n_atoms} atoms in a "
        f"{crystal_system} cell (space group {space_group}). The DFT "
        f"flag is_metal = {is_metal}; the band-gap class given the "
        f"cell is {bg_class}.\n"
        f"{chemistry_note}"
    ).strip("\n")


def _probe_consistency_review(
    probe_outputs: dict[str, dict[str, Any]],
    ground_truth: GroundTruthMaterials,
) -> str:
    """Step 3: confidence-aware per-probe review."""
    lines: list[str] = []
    e_form = _read(probe_outputs, "formation_energy")
    e_hull = _read(probe_outputs, "e_above_hull")
    bg = _read(probe_outputs, "band_gap")
    is_metal = _read(probe_outputs, "is_metal")
    sg = _read(probe_outputs, "space_group")

    e_form_pred = e_form.get("prediction")
    e_form_true = ground_truth.get("formation_energy")
    if e_form_pred is not None and e_form_true is not None:
        delta = abs(float(e_form_pred) - float(e_form_true))
        lines.append(
            f"  - formation_energy probe : "
            f"{float(e_form_pred):.3f} eV/atom "
            f"(confidence {_confidence(e_form):.2f}); "
            f"deviation from truth = {delta:.3f}."
        )

    e_hull_pred = e_hull.get("prediction")
    e_hull_true = ground_truth.get("e_above_hull")
    is_stable_truth = bool(ground_truth.get("is_stable"))
    if e_hull_pred is not None and e_hull_true is not None:
        threshold_dist = abs(float(e_hull_pred) - 0.025)
        lines.append(
            f"  - e_above_hull probe     : "
            f"{float(e_hull_pred):.3f} eV/atom "
            f"(confidence {_confidence(e_hull):.2f}); "
            f"distance from 0.025 stability threshold = {threshold_dist:.3f}; "
            f"truth is_stable = {is_stable_truth}."
        )

    bg_pred = bg.get("prediction")
    is_metal_pred = is_metal.get("prediction")
    if bg_pred is not None:
        lines.append(
            f"  - band_gap probe         : "
            f"{float(bg_pred):.2f} eV "
            f"(confidence {_confidence(bg):.2f})."
        )
    if is_metal_pred is not None:
        lines.append(
            f"  - is_metal probe         : "
            f"{is_metal_pred} "
            f"(confidence {_confidence(is_metal):.2f}); this is the "
            f"authoritative metal/non-metal signal -- defer to it for "
            f"band_gap_class when confidence is high."
        )
    if sg.get("prediction") is not None:
        lines.append(
            f"  - space_group probe      : #{sg['prediction']} "
            f"(confidence {_confidence(sg):.2f})."
        )
    return "\n".join(lines)


def _counterfactual_check(
    probe_outputs: dict[str, dict[str, Any]],
    ground_truth: GroundTruthMaterials,
) -> str:
    """Step 4: a brief counterfactual sanity check."""
    bg = _read(probe_outputs, "band_gap")
    is_metal = _read(probe_outputs, "is_metal")
    bg_pred = bg.get("prediction")
    is_metal_pred = str(is_metal.get("prediction") or "").lower()
    bg_class_truth = ground_truth.get("band_gap_class") or ""

    if bg_pred is None:
        return "  (no counterfactual possible without band-gap probe)"

    bg_val = float(bg_pred)
    if is_metal_pred == "metal" and bg_val > 0.5:
        return (
            f"  Tension: is_metal probe says 'metal' but band_gap probe "
            f"predicts {bg_val:.2f} eV (a non-zero gap). High is_metal "
            f"confidence trumps small-gap predictions. Truth says "
            f"'{bg_class_truth}'."
        )
    if is_metal_pred == "non_metal" and bg_val <= 0.05:
        return (
            f"  Tension: is_metal probe says 'non_metal' but band_gap "
            f"probe predicts ~0 eV. Probe MAE is on the order of 0.5 "
            f"eV; non-metal status is more reliable than the "
            f"sub-0.05-eV band-gap value. Truth says '{bg_class_truth}'."
        )
    return (
        f"  Probes are mutually consistent: is_metal = {is_metal_pred}, "
        f"band_gap = {bg_val:.2f} eV. The is_metal classification "
        f"determines whether the band_gap_class is 'metal'; non-metals "
        f"split into 'narrow' (<= 3 eV) vs 'wide' (> 3 eV) by the "
        f"band_gap probe value. Truth says '{bg_class_truth}'."
    )


def generate_rich_cot(
    *,
    probe_outputs: dict[str, dict[str, Any]],
    ground_truth: GroundTruthMaterials,
    sae_features: list[tuple[str, float]] | None = None,
    feature_metadata: dict | None = None,
) -> MaterialsCoT:
    """Rich materials CoT (v2): probes + SAE + composition + counterfactual.

    Layout:
      Step 1     - read probes (5 lines)
      Step 1b    - SAE features with calibrated activation strength and
                   representative-specimen grounding
      Step 2     - compositional sanity check (formula, lattice, priors)
      Step 3     - probe consistency review with confidence
      Step 4     - counterfactual / probe-tension resolution
      Final commit - ground-truth JSON

    The Final commit JSON still comes from ground_truth (training
    signal). Steps 1-4 are scaffolding that gives the LLM a richer
    chain to learn during SFT.
    """
    cot = generate_cot(
        probe_outputs=probe_outputs,
        ground_truth=ground_truth,
        sae_features=sae_features,
    )
    consistent = cot.consistent
    final_claim = cot.final_claim

    lines: list[str] = []

    # Step 1: probes (reuse compact rendering).
    e_form_pred = _read(probe_outputs, "formation_energy")
    e_hull_pred = _read(probe_outputs, "e_above_hull")
    bg_pred = _read(probe_outputs, "band_gap")
    is_metal_pred_dict = _read(probe_outputs, "is_metal")
    sg_pred = _read(probe_outputs, "space_group")
    bg_value = float(bg_pred.get("prediction") or 0.0)
    is_metal_value = (
        str(is_metal_pred_dict.get("prediction") or "").lower() or None
    )

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
            f"{float(e_hull_pred['prediction']):.3f} eV/atom "
            f"(confidence {_confidence(e_hull_pred):.2f})"
        )
    if bg_pred["prediction"] is not None:
        lines.append(
            f"  - band-gap probe         : "
            f"{bg_value:.2f} eV "
            f"(confidence {_confidence(bg_pred):.2f})"
        )
    if is_metal_pred_dict["prediction"] is not None:
        lines.append(
            f"  - is-metal probe         : "
            f"{is_metal_value} "
            f"(confidence {_confidence(is_metal_pred_dict):.2f})"
        )
    if sg_pred["prediction"] is not None:
        lines.append(
            f"  - space-group probe      : "
            f"#{sg_pred['prediction']} "
            f"(confidence {_confidence(sg_pred):.2f})"
        )

    # Step 1b: rich SAE features.
    if sae_features:
        lines.append("")
        lines.append(
            "Step 1b - SAE-derived structural / electronic context:"
        )
        lines.append(
            _format_rich_sae_features(sae_features, feature_metadata)
        )

    # Step 2: compositional sanity check.
    lines.append("")
    lines.append("Step 2 - Compositional sanity check:")
    lines.append(_compositional_sanity(ground_truth))

    # Step 3: probe consistency review.
    lines.append("")
    lines.append("Step 3 - Probe consistency review:")
    lines.append(_probe_consistency_review(probe_outputs, ground_truth))

    # Step 4: counterfactual / tension check.
    lines.append("")
    lines.append("Step 4 - Counterfactual / tension check:")
    lines.append(_counterfactual_check(probe_outputs, ground_truth))

    # Final commit.
    lines.append("")
    lines.append(f"Final commit: {json.dumps(final_claim, sort_keys=True)}")

    return MaterialsCoT(
        text="\n".join(lines),
        consistent=consistent,
        final_claim=final_claim,
    )


def build_rich_sft_record(
    *,
    probe_outputs: dict[str, dict[str, Any]],
    ground_truth: GroundTruthMaterials,
    specimen_id: int,
    sae_features: list[tuple[str, float]] | None = None,
    feature_metadata: dict | None = None,
) -> dict[str, Any]:
    """Rich-CoT analogue of build_sft_record."""
    cot = generate_rich_cot(
        probe_outputs=probe_outputs,
        ground_truth=ground_truth,
        sae_features=sae_features,
        feature_metadata=feature_metadata,
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
        "cot_version": "v2_rich",
    }


__all__ = [
    "GroundTruthMaterials",
    "MaterialsCoT",
    "band_gap_consistent",
    "build_rich_sft_record",
    "build_sft_record",
    "generate_cot",
    "generate_rich_cot",
    "stability_consistent",
]
