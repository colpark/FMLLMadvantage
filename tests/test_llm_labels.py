"""Tests for Phase 15 Stage C labelling.

CPU-only. Cover:

  - Constructed activations whose top activators are all wrong PASS
    rows yield a label with verdict=pass and correct=False at high
    purity.
  - Activations whose top activators concentrate on one motif yield
    a motif lock.
  - Continuous correlations with atom_counts and temperatures fire
    when |r| >= min_corr.
  - rank_features_for_steering returns the right candidates for
    target_axis='correct', target_value=False.
"""

from __future__ import annotations

import numpy as np

from fmllm.representation.llm_labels import (
    LLMFeatureLabel,
    label_llm_feature,
    rank_features_for_steering,
)


def _attrs(n: int):
    """Return zero-filled attribute arrays of length n."""
    return {
        "verdicts": np.array(["pass"] * n),
        "is_correct": np.array([True] * n),
        "motifs": np.array(["ring"] * n),
        "phases": np.array(["solid-like"] * n),
        "atom_counts": np.zeros(n, dtype=np.float32),
        "temperatures": np.zeros(n, dtype=np.float32),
    }


def test_label_llm_feature_locks_on_wrong_pass_rows():
    rng = np.random.default_rng(0)
    n = 100
    feat = rng.uniform(0.0, 0.1, size=n).astype(np.float32)
    # Make the top 50 activators be wrong PASS rows specifically.
    top_idx = np.arange(50)
    feat[top_idx] = rng.uniform(1.0, 2.0, size=50)
    attrs = _attrs(n)
    attrs["verdicts"] = np.array(
        ["pass"] * 50 + ["caveat"] * 50,
    )
    attrs["is_correct"] = np.array([False] * 50 + [True] * 50)
    rec = label_llm_feature(
        feature_idx=42,
        feature_activations=feat,
        top_n=50, min_purity=0.70, min_corr=0.30,
        **attrs,
    )
    assert rec.verdict_top == "pass"
    assert rec.correct_top is False
    assert rec.verdict_purity is not None and rec.verdict_purity >= 0.70
    assert rec.correct_purity is not None and rec.correct_purity >= 0.70
    assert "verdict=pass" in rec.label
    assert "correct=False" in rec.label


def test_label_llm_feature_continuous_correlation_with_n_atoms():
    n = 100
    # Activation that grows with atom count.
    n_atoms = np.linspace(5, 30, n).astype(np.float32)
    feat = (n_atoms - n_atoms.mean()).astype(np.float32)
    feat[feat < 0] = 0.0           # nonzero on the high-N half
    attrs = _attrs(n)
    attrs["atom_counts"] = n_atoms
    rec = label_llm_feature(
        feature_idx=7,
        feature_activations=feat,
        top_n=50, min_purity=0.70, min_corr=0.30,
        **attrs,
    )
    assert rec.n_atoms_corr is not None and abs(rec.n_atoms_corr) >= 0.30
    assert any("N-" in t for t in rec.tags)


def test_label_llm_feature_falls_back_when_rare():
    n = 100
    feat = np.zeros(n, dtype=np.float32)
    feat[:3] = 1.0                 # only 3 nonzero rows
    attrs = _attrs(n)
    rec = label_llm_feature(
        feature_idx=99,
        feature_activations=feat,
        top_n=50, min_purity=0.70, min_corr=0.30,
        **attrs,
    )
    assert "rare" in rec.label
    assert rec.verdict_top is None
    assert rec.correct_top is None


def test_label_llm_feature_motif_lock():
    rng = np.random.default_rng(0)
    n = 100
    feat = rng.uniform(0.0, 0.1, size=n).astype(np.float32)
    feat[0:50] = rng.uniform(1.0, 2.0, size=50)
    motifs = np.array(["triangular_disk"] * 50 + ["ring"] * 50)
    phases = np.array(["solid-like"] * n)
    rec = label_llm_feature(
        feature_idx=11,
        feature_activations=feat,
        verdicts=np.array(["pass"] * n),
        is_correct=np.array([True] * n),
        motifs=motifs, phases=phases,
        atom_counts=np.zeros(n, dtype=np.float32),
        temperatures=np.zeros(n, dtype=np.float32),
        top_n=50, min_purity=0.70, min_corr=0.30,
    )
    assert rec.motif_top == "triangular_disk"
    assert "motif=triangular_disk" in rec.label


def _mk_label(idx: int, verdict_top, verdict_purity,
              correct_top, correct_purity, n_top=50) -> LLMFeatureLabel:
    return LLMFeatureLabel(
        feature_idx=idx,
        label=f"f{idx}",
        verdict_top=verdict_top,
        verdict_purity=verdict_purity,
        correct_top=correct_top,
        correct_purity=correct_purity,
        n_top_activators=n_top,
    )


def test_rank_for_steering_returns_wrong_features_in_purity_order():
    labels = [
        _mk_label(0, "pass", 0.80, False, 0.85),       # passes (correct=False)
        _mk_label(1, "pass", 0.80, False, 0.95),       # passes, higher purity
        _mk_label(2, "pass", 0.80, True,  0.90),       # correct=True, drop
        _mk_label(3, "pass", 0.80, False, 0.65),       # below min_purity, drop
        _mk_label(4, None,   None,  None,  None),       # no lock, drop
    ]
    out = rank_features_for_steering(
        labels, target_axis="correct", target_value=False, min_purity=0.70,
    )
    assert [l.feature_idx for l in out] == [1, 0]


def test_rank_for_steering_for_caveat_verdict():
    labels = [
        _mk_label(0, "caveat", 0.95, True, 0.50),     # passes (caveat)
        _mk_label(1, "pass",   0.95, True, 0.50),     # not caveat, drop
        _mk_label(2, "caveat", 0.65, True, 0.50),     # below threshold, drop
    ]
    out = rank_features_for_steering(
        labels, target_axis="verdict", target_value="caveat", min_purity=0.70,
    )
    assert [l.feature_idx for l in out] == [0]
