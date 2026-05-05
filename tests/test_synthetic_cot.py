"""Tests for the Phase 11 probe bank + synthetic CoT generator.

CPU-only. Cover:

  - ProbeBank save/load round trip with both regression and
    classification probes.
  - ProbeBank.evaluate returns the documented dict shape.
  - expected_coordination matches physical intuition for the three
    motifs.
  - generate_cot is deterministic, mentions probe outputs, and
    commits the ground-truth claim verbatim.
  - build_sft_record emits the (system, user, assistant) chat
    structure Phase 6's train_sft consumes.
"""

from __future__ import annotations

import json

import torch

from fmllm.training.probe_bank import ProbeBank, ProbeSpec, _build_module
from fmllm.training.synthetic_cot import (
    build_sft_record,
    coordination_consistent,
    expected_coordination,
    generate_cot,
)


# ---------------------------------------------------------------------------
# Probe bank
# ---------------------------------------------------------------------------


def _toy_probe_bank() -> ProbeBank:
    bank = ProbeBank()
    spec_n = ProbeSpec(
        name="n_atoms", kind="regression", in_dim=8, out_dim=1, hidden=16,
        target_min=5.0, target_max=30.0, target_mean=15.0, target_std=7.0,
    )
    bank.add(spec_n, _build_module(spec_n))
    spec_motif = ProbeSpec(
        name="motif", kind="classification", in_dim=8, out_dim=3, hidden=16,
        class_names=["triangular_disk", "ring", "linear"],
    )
    bank.add(spec_motif, _build_module(spec_motif))
    return bank


def test_probe_bank_evaluate_shape():
    bank = _toy_probe_bank().eval()
    features = torch.randn(4, 8)
    out = bank.evaluate(features)
    assert isinstance(out, list)
    assert len(out) == 4
    for row in out:
        assert "n_atoms" in row
        assert "motif" in row
        assert row["n_atoms"]["kind"] == "regression"
        assert row["motif"]["kind"] == "classification"
        assert isinstance(row["n_atoms"]["prediction"], float)
        assert row["motif"]["prediction"] in {"triangular_disk", "ring", "linear"}
        assert "class_probs" in row["motif"]


def test_probe_bank_save_load_round_trip(tmp_path):
    bank = _toy_probe_bank().eval()
    out_dir = bank.save(tmp_path / "probes")
    reloaded = ProbeBank.load(out_dir)
    assert sorted(reloaded.names()) == sorted(bank.names())
    # Apply both to the same input and confirm bit-identical output.
    features = torch.randn(2, 8)
    a = bank.eval().evaluate(features)
    b = reloaded.eval().evaluate(features)
    for row_a, row_b in zip(a, b, strict=True):
        assert row_a["motif"]["prediction"] == row_b["motif"]["prediction"]
        assert abs(row_a["n_atoms"]["prediction"] - row_b["n_atoms"]["prediction"]) < 1.0e-6


# ---------------------------------------------------------------------------
# expected_coordination
# ---------------------------------------------------------------------------


def test_expected_coordination_ring_is_two():
    assert expected_coordination(7, "ring") == 2.0
    assert expected_coordination(20, "ring") == 2.0


def test_expected_coordination_linear_endpoint_correction():
    # linear chain of N atoms: 2 endpoints with 1 neighbor, N-2 interior
    # with 2 neighbors. mean = 2(N-1)/N.
    assert abs(expected_coordination(2, "linear") - 1.0) < 1.0e-6
    assert abs(expected_coordination(10, "linear") - 1.8) < 1.0e-6


def test_expected_coordination_triangular_disk_grows_with_N():
    a = expected_coordination(7, "triangular_disk")
    b = expected_coordination(20, "triangular_disk")
    assert a < b   # bigger disks have more interior atoms = higher coord


def test_coordination_consistent_thresholding():
    # 11-atom triangular_disk: expected ~3.34. Observed 3.6 should be
    # consistent under tolerance 0.6.
    ok, expected, diff = coordination_consistent(3.6, 11, "triangular_disk")
    assert ok
    assert abs(expected - 3.34) < 0.05
    assert diff < 0.6
    # Observed 6.0 is way off; should fail.
    ok2, _, _ = coordination_consistent(6.0, 11, "triangular_disk")
    assert ok2 is False


# ---------------------------------------------------------------------------
# Synthetic CoT generator
# ---------------------------------------------------------------------------


def _example_probes() -> dict[str, dict[str, object]]:
    return {
        "n_atoms": {"prediction": 11.2, "confidence": 0.86, "kind": "regression"},
        "motif": {"prediction": "triangular_disk", "confidence": 0.79, "kind": "classification"},
        "phase": {"prediction": "solid-like", "confidence": 0.92, "kind": "classification"},
        "coordination": {"prediction": 3.4, "confidence": 0.7, "kind": "regression"},
        "peak_position": {"prediction": 1.13, "confidence": 0.8, "kind": "regression"},
    }


def test_generate_cot_is_deterministic():
    truth = {"n": 11, "motif": "triangular_disk", "t": 0.20}
    a = generate_cot(probe_outputs=_example_probes(), ground_truth=truth)
    b = generate_cot(probe_outputs=_example_probes(), ground_truth=truth)
    assert a.text == b.text
    assert a.consistent == b.consistent


def test_generate_cot_mentions_each_probe():
    truth = {"n": 11, "motif": "triangular_disk", "t": 0.20}
    cot = generate_cot(probe_outputs=_example_probes(), ground_truth=truth)
    assert "atom-count probe" in cot.text
    assert "motif probe" in cot.text
    assert "phase probe" in cot.text
    assert "coordination" in cot.text
    assert "RDF first peak" in cot.text


def test_generate_cot_commits_ground_truth_verbatim():
    """The final commit must come from ground truth, not from the
    probes. This is what teaches the LLM that probes are inputs and
    truth is the output."""
    truth = {"n": 11, "motif": "triangular_disk", "t": 0.20}
    probes = _example_probes()
    # Deliberately make the n_atoms probe wildly wrong to exercise this
    probes["n_atoms"]["prediction"] = 25.0
    cot = generate_cot(probe_outputs=probes, ground_truth=truth)
    final = cot.final_claim
    assert final["n_atoms"] == 11
    assert final["motif"] == "triangular_disk"
    assert abs(final["temperature"] - 0.20) < 1.0e-6
    # And the rendered text should still reference the wrong probe value
    # (i.e. the CoT preserves the disagreement rather than hiding it).
    assert "25.0" in cot.text


def test_generate_cot_inconsistent_branch():
    truth = {"n": 7, "motif": "ring", "t": 0.50}
    probes = _example_probes()
    probes["motif"] = {
        "prediction": "linear", "confidence": 0.40, "kind": "classification",
    }
    cot = generate_cot(probe_outputs=probes, ground_truth=truth)
    # Coord 3.4 is far from linear N=11's expected 1.82, so probes
    # should be flagged inconsistent and the resolution branch should
    # mention the highest-confidence probe.
    assert cot.consistent is False
    assert (
        "Defer to the highest-confidence probe"
        in cot.text
    )


def test_build_sft_record_shape():
    truth = {"n": 11, "motif": "triangular_disk", "t": 0.20}
    record = build_sft_record(
        probe_outputs=_example_probes(),
        ground_truth=truth,
        specimen_id=42,
    )
    assert record["specimen_id"] == 42
    assert isinstance(record["messages"], list)
    assert len(record["messages"]) == 3
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    # The assistant content must be the rendered CoT text.
    assert "Step 1 - Read the probes:" in record["messages"][2]["content"]
    # User message must include the probes as JSON so the LLM at
    # inference time can parse them.
    user_text = record["messages"][1]["content"]
    assert "PROBES" in user_text
    payload_start = user_text.index("{")
    payload_end = user_text.rindex("}") + 1
    payload = json.loads(user_text[payload_start:payload_end])
    assert "n_atoms" in payload
    assert "motif" in payload
