"""CPU tests for fmllm.materials.labels."""

from __future__ import annotations

import numpy as np

from fmllm.materials.labels import label_materials_feature


def _attrs(n: int):
    return {
        "crystal_systems": np.array(["cubic"] * n),
        "is_metals": np.zeros(n, dtype=bool),
        "band_gap_classes": np.array(["narrow"] * n),
        "formation_energies": np.zeros(n, dtype=np.float32),
        "e_above_hulls": np.zeros(n, dtype=np.float32),
        "band_gaps": np.zeros(n, dtype=np.float32),
        "n_atoms": np.zeros(n, dtype=np.float32),
    }


def test_locks_on_dominant_crystal_system():
    rng = np.random.default_rng(0)
    n = 100
    feat = rng.uniform(0.0, 0.1, size=n).astype(np.float32)
    feat[0:50] = rng.uniform(1.0, 2.0, size=50)
    attrs = _attrs(n)
    attrs["crystal_systems"] = np.array(["cubic"] * 50 + ["hexagonal"] * 50)
    rec = label_materials_feature(
        feature_idx=42,
        feature_activations=feat,
        top_n=50,
        min_purity=0.70,
        min_corr=0.30,
        **attrs,
    )
    assert rec.crystal_system_top == "cubic"
    assert "crystal=cubic" in rec.label


def test_continuous_correlation_with_e_form():
    n = 100
    e_form = np.linspace(-3.0, 0.0, n).astype(np.float32)
    feat = (-(e_form - e_form.mean())).astype(np.float32)
    feat[feat < 0] = 0.0
    attrs = _attrs(n)
    attrs["formation_energies"] = e_form
    rec = label_materials_feature(
        feature_idx=7,
        feature_activations=feat,
        top_n=50,
        min_purity=0.70,
        min_corr=0.30,
        **attrs,
    )
    assert rec.formation_energy_corr is not None
    assert abs(rec.formation_energy_corr) >= 0.30
    assert any("e_form" in t for t in rec.tags)


def test_falls_back_when_rare():
    n = 100
    feat = np.zeros(n, dtype=np.float32)
    feat[:3] = 1.0
    attrs = _attrs(n)
    rec = label_materials_feature(
        feature_idx=99,
        feature_activations=feat,
        top_n=50,
        min_purity=0.70,
        min_corr=0.30,
        **attrs,
    )
    assert "rare" in rec.label
    assert rec.crystal_system_top is None


def test_band_gap_class_lock():
    rng = np.random.default_rng(0)
    n = 100
    feat = rng.uniform(0.0, 0.1, size=n).astype(np.float32)
    feat[0:50] = rng.uniform(1.0, 2.0, size=50)
    attrs = _attrs(n)
    attrs["band_gap_classes"] = np.array(["wide"] * 50 + ["metal"] * 50)
    rec = label_materials_feature(
        feature_idx=11,
        feature_activations=feat,
        top_n=50,
        min_purity=0.70,
        min_corr=0.30,
        **attrs,
    )
    assert rec.band_gap_class_top == "wide"
    assert "gap_class=wide" in rec.label
