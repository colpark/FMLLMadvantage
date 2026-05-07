"""Correlation-based labelling of SAE features for the materials port.

Mirrors ``fmllm.representation.labels`` but with materials-specific
attribute names. The labelling recipe is identical: for each SAE
feature, find the top-N activating specimens, check whether they
concentrate on a single category at >= ``min_purity``, and tag
continuous attributes whose Pearson correlation exceeds ``min_corr``.

Materials-side categorical axes:
    crystal_system : "cubic" / "hexagonal" / ... (7 systems)
    is_metal       : True / False
    band_gap_class : "metal" / "narrow" / "wide"

Materials-side continuous axes:
    formation_energy : eV/atom
    e_above_hull     : eV/atom
    band_gap         : eV
    n_atoms          : count
    total_magnetization : μB

Depends on:
    numpy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MaterialsFeatureLabel:
    """One feature's structured label record (materials version)."""

    feature_idx: int
    label: str
    crystal_system_top: str | None = None
    crystal_system_purity: float | None = None
    is_metal_top: bool | None = None
    is_metal_purity: float | None = None
    band_gap_class_top: str | None = None
    band_gap_class_purity: float | None = None
    formation_energy_corr: float | None = None
    e_above_hull_corr: float | None = None
    band_gap_corr: float | None = None
    n_atoms_corr: float | None = None
    n_top_activators: int = 0
    activation_mean_top: float = 0.0
    tags: list[str] = field(default_factory=list)


def _categorical_lock(values: list, min_purity: float) -> tuple[object | None, float]:
    if not values:
        return None, 0.0
    counts = Counter(values)
    most, n = counts.most_common(1)[0]
    purity = n / len(values)
    if purity >= min_purity:
        return most, purity
    return None, purity


def _continuous_corr(activations: np.ndarray, attribute: np.ndarray) -> float:
    if activations.shape[0] < 3:
        return 0.0
    if activations.std() < 1.0e-8 or attribute.std() < 1.0e-8:
        return 0.0
    return float(np.corrcoef(activations, attribute)[0, 1])


def label_materials_feature(
    *,
    feature_idx: int,
    feature_activations: np.ndarray,
    crystal_systems: np.ndarray,
    is_metals: np.ndarray,
    band_gap_classes: np.ndarray,
    formation_energies: np.ndarray,
    e_above_hulls: np.ndarray,
    band_gaps: np.ndarray,
    n_atoms: np.ndarray,
    top_n: int = 50,
    min_purity: float = 0.70,
    min_corr: float = 0.30,
) -> MaterialsFeatureLabel:
    """Build a label for one SAE feature using materials attributes."""
    n_specimens = int(feature_activations.shape[0])
    if n_specimens == 0:
        return MaterialsFeatureLabel(
            feature_idx=feature_idx, label=f"f{feature_idx}",
        )

    nonzero = feature_activations > 1.0e-6
    n_nonzero = int(nonzero.sum())
    if n_nonzero < max(5, top_n // 5):
        return MaterialsFeatureLabel(
            feature_idx=feature_idx,
            label=f"f{feature_idx} (rare)",
            n_top_activators=n_nonzero,
        )

    take = min(top_n, n_nonzero)
    top_idx = np.argsort(feature_activations)[::-1][:take]

    top_cs = [str(c) for c in crystal_systems[top_idx].tolist()]
    top_ism = [bool(b) for b in is_metals[top_idx].tolist()]
    top_bgc = [str(c) for c in band_gap_classes[top_idx].tolist()]
    top_acts = feature_activations[top_idx]

    cs_top, cs_purity = _categorical_lock(top_cs, min_purity)
    ism_top, ism_purity = _categorical_lock(top_ism, min_purity)
    bgc_top, bgc_purity = _categorical_lock(top_bgc, min_purity)

    e_form_corr = _continuous_corr(
        feature_activations, formation_energies.astype(np.float64),
    )
    e_hull_corr = _continuous_corr(
        feature_activations, e_above_hulls.astype(np.float64),
    )
    bg_corr = _continuous_corr(
        feature_activations, band_gaps.astype(np.float64),
    )
    n_corr = _continuous_corr(
        feature_activations, n_atoms.astype(np.float64),
    )

    tags: list[str] = []
    if cs_top is not None:
        tags.append(f"crystal={cs_top}")
    if ism_top is not None:
        tags.append(f"metal={ism_top}")
    if bgc_top is not None and bgc_top != "":
        tags.append(f"gap_class={bgc_top}")
    if abs(e_form_corr) >= min_corr:
        direction = "high" if e_form_corr > 0 else "low"
        tags.append(f"e_form-{direction}(r={e_form_corr:+.2f})")
    if abs(e_hull_corr) >= min_corr:
        direction = "high" if e_hull_corr > 0 else "low"
        tags.append(f"hull-{direction}(r={e_hull_corr:+.2f})")
    if abs(bg_corr) >= min_corr:
        direction = "wide" if bg_corr > 0 else "narrow"
        tags.append(f"gap-{direction}(r={bg_corr:+.2f})")
    if abs(n_corr) >= min_corr:
        direction = "many" if n_corr > 0 else "few"
        tags.append(f"natoms-{direction}(r={n_corr:+.2f})")

    label = (
        f"f{feature_idx}: " + " + ".join(tags)
        if tags
        else f"f{feature_idx}: unlabelled (no significant pattern)"
    )

    return MaterialsFeatureLabel(
        feature_idx=feature_idx,
        label=label,
        crystal_system_top=cs_top if isinstance(cs_top, str) else None,
        crystal_system_purity=cs_purity if isinstance(cs_top, str) else None,
        is_metal_top=ism_top if isinstance(ism_top, bool) else None,
        is_metal_purity=ism_purity if isinstance(ism_top, bool) else None,
        band_gap_class_top=bgc_top if isinstance(bgc_top, str) else None,
        band_gap_class_purity=bgc_purity if isinstance(bgc_top, str) else None,
        formation_energy_corr=e_form_corr,
        e_above_hull_corr=e_hull_corr,
        band_gap_corr=bg_corr,
        n_atoms_corr=n_corr,
        n_top_activators=int(take),
        activation_mean_top=float(top_acts.mean()),
        tags=tags,
    )


__all__ = ["MaterialsFeatureLabel", "label_materials_feature"]
