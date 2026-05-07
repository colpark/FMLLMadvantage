"""Tests for the materials synthetic CoT generator (mirror of LJ tests).

CPU-only. Cover:

  - generate_cot is deterministic given the same inputs.
  - generate_cot mentions every probe by name in Step 1.
  - generate_cot's final commit comes from ground truth, not probes.
  - SAE features render Step 1b when supplied.
  - The user message contains both PROBES and SAE_FEATURES payloads.
  - Stability and band-gap consistency helpers behave as documented.
  - is_correct enforces the joint correctness criterion.
"""

from __future__ import annotations

import json

from fmllm.materials.ground_truth import (
    band_gap_class,
    is_correct,
)
from fmllm.materials.synthetic_cot import (
    band_gap_consistent,
    build_sft_record,
    generate_cot,
    stability_consistent,
)


def _example_probes() -> dict:
    return {
        "formation_energy": {"prediction": -2.85, "confidence": 0.91},
        "e_above_hull": {"prediction": 0.018, "confidence": 0.85},
        "band_gap": {"prediction": 1.23, "confidence": 0.88},
        "is_metal": {"prediction": "non_metal", "confidence": 0.93},
        "space_group": {"prediction": 225, "confidence": 0.79},
    }


def _example_truth() -> dict:
    return {
        "formation_energy": -2.85,
        "e_above_hull": 0.018,
        "is_stable": True,
        "band_gap": 1.23,
        "band_gap_class": "narrow",
        "space_group": 225,
        "crystal_system": "cubic",
        "is_metal": False,
        "total_magnetization": 0.0,
        "n_atoms": 4,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_band_gap_class():
    assert band_gap_class(0.0) == "metal"
    assert band_gap_class(1.5) == "narrow"
    assert band_gap_class(5.0) == "wide"
    assert band_gap_class(0.0, is_metal=True) == "metal"


def test_stability_consistent_agrees():
    ok, msg = stability_consistent(e_above_hull=0.01, is_stable_pred=True)
    assert ok
    assert "below" in msg.lower()


def test_stability_consistent_disagrees():
    ok, msg = stability_consistent(e_above_hull=0.05, is_stable_pred=True)
    assert not ok
    assert "disagree" in msg.lower()


def test_band_gap_consistent_metal():
    ok, _ = band_gap_consistent(0.0, "metal")
    assert ok


def test_band_gap_consistent_wrong_class():
    ok, _ = band_gap_consistent(2.0, "wide")
    assert not ok


# ---------------------------------------------------------------------------
# generate_cot
# ---------------------------------------------------------------------------


def test_generate_cot_is_deterministic():
    a = generate_cot(probe_outputs=_example_probes(), ground_truth=_example_truth())
    b = generate_cot(probe_outputs=_example_probes(), ground_truth=_example_truth())
    assert a.text == b.text
    assert a.final_claim == b.final_claim


def test_generate_cot_mentions_each_probe():
    cot = generate_cot(
        probe_outputs=_example_probes(), ground_truth=_example_truth(),
    )
    txt = cot.text
    assert "formation-energy probe" in txt
    assert "e-above-hull probe" in txt
    assert "band-gap probe" in txt
    assert "is-metal probe" in txt
    assert "space-group probe" in txt
    assert "Step 1 - Read the probes" in txt
    assert "Step 2" in txt
    assert "Step 3" in txt
    assert "Final commit" in txt


def test_generate_cot_commits_ground_truth():
    truth = _example_truth()
    truth["space_group"] = 99       # unusual value
    truth["band_gap_class"] = "narrow"
    cot = generate_cot(
        probe_outputs=_example_probes(), ground_truth=truth,
    )
    assert cot.final_claim["space_group"] == 99
    assert cot.final_claim["band_gap_class"] == "narrow"
    # Ground truth dictates the commit even though the probe says 225.
    assert "99" in cot.text.split("Final commit:")[-1]


def test_generate_cot_with_sae_features_renders_step_1b():
    sae = [
        ("f127: crystal=cubic + e_form-low(r=-0.62)", 3.04),
        ("f342: e_above_hull-low (purity 0.92)", 2.81),
    ]
    cot = generate_cot(
        probe_outputs=_example_probes(),
        ground_truth=_example_truth(),
        sae_features=sae,
    )
    assert "Step 1b" in cot.text
    assert "f127" in cot.text
    assert "f342" in cot.text


def test_generate_cot_without_sae_features_omits_step_1b():
    cot = generate_cot(
        probe_outputs=_example_probes(), ground_truth=_example_truth(),
    )
    assert "Step 1b" not in cot.text


# ---------------------------------------------------------------------------
# build_sft_record
# ---------------------------------------------------------------------------


def test_build_sft_record_shape():
    record = build_sft_record(
        probe_outputs=_example_probes(),
        ground_truth=_example_truth(),
        specimen_id=42,
    )
    assert record["specimen_id"] == 42
    assert len(record["messages"]) == 3
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert "Step 1 - Read the probes" in record["messages"][2]["content"]
    user_text = record["messages"][1]["content"]
    assert "PROBES" in user_text
    assert "formation_energy" in user_text


def test_build_sft_record_with_sae_appears_in_user():
    sae = [("f10: cubic + non-metal", 1.5)]
    record = build_sft_record(
        probe_outputs=_example_probes(),
        ground_truth=_example_truth(),
        specimen_id=42,
        sae_features=sae,
    )
    user_text = record["messages"][1]["content"]
    assert "PROBES" in user_text
    assert "SAE_FEATURES" in user_text
    assert "f10" in user_text
    assert record["sae_features_count"] == 1


# ---------------------------------------------------------------------------
# is_correct
# ---------------------------------------------------------------------------


def test_is_correct_strict():
    truth = _example_truth()
    claim_right = {
        "formation_energy": -2.85,
        "e_above_hull": 0.018,
        "is_stable": True,
        "band_gap_class": "narrow",
        "space_group": 225,
    }
    assert is_correct(claim_right, truth)

    # tolerance: formation_energy off by 0.04 is OK (<= 0.05)
    claim_close = dict(claim_right)
    claim_close["formation_energy"] = -2.81
    assert is_correct(claim_close, truth)

    # tolerance: formation_energy off by 0.06 is NOT OK
    claim_too_far = dict(claim_right)
    claim_too_far["formation_energy"] = -2.79
    assert not is_correct(claim_too_far, truth)

    # band_gap_class mismatch
    claim_wrong_class = dict(claim_right)
    claim_wrong_class["band_gap_class"] = "wide"
    assert not is_correct(claim_wrong_class, truth)

    # space_group mismatch
    claim_wrong_sg = dict(claim_right)
    claim_wrong_sg["space_group"] = 1
    assert not is_correct(claim_wrong_sg, truth)
