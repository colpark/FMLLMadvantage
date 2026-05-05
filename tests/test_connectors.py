"""Tests for the Phase 9 connector pieces.

These tests run on a CPU and need only torch + numpy. They cover:

* Q-Former forward shape and parameter trainability.
* Templated annotation determinism, content faithfulness, and the
  positions-optional path.
* FM2.encode shape and CLS-token contract (forward calls into encode
  and produces the same energy).
* Decision-rule helpers in the probing CLI (no GPU work).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fmllm.connectors import FM2Connector, annotate_specimen
from fmllm.connectors.text_annotations import (
    SpecimenAnnotation,
    annotation_label_dict,
)
from fmllm.fms.fm2_rdf.model import FM2RDFTransformer


# ---------------------------------------------------------------------------
# FM2.encode contract
# ---------------------------------------------------------------------------


def test_fm2_encode_returns_full_sequence():
    model = FM2RDFTransformer(rdf_bins=200, embed_dim=64, depth=2, num_heads=4)
    rdf = torch.randn(3, 200)
    out = model.encode(rdf)
    assert out.shape == (3, 201, 64)


def test_fm2_forward_uses_encode_path():
    """forward() must reach the same energy whether we call it directly
    or compose encode() + head manually. This is the contract probes
    and the connector rely on."""
    torch.manual_seed(0)
    model = FM2RDFTransformer(rdf_bins=200, embed_dim=64, depth=2, num_heads=4)
    model.eval()
    rdf = torch.randn(2, 200)
    with torch.no_grad():
        e_direct = model(rdf)
        hidden = model.encode(rdf)
        e_composed = model.energy_head(hidden[:, 0]).squeeze(-1)
    assert torch.allclose(e_direct, e_composed, atol=1.0e-6)


# ---------------------------------------------------------------------------
# Q-Former
# ---------------------------------------------------------------------------


def test_qformer_forward_shape():
    conn = FM2Connector(
        fm_dim=64, llm_dim=128, n_query=8, n_layers=2, n_heads=4,
    )
    fm = torch.randn(3, 201, 64)
    out = conn(fm)
    assert out.shape == (3, 8, 128)


def test_qformer_only_connector_has_grad():
    conn = FM2Connector(
        fm_dim=64, llm_dim=128, n_query=4, n_layers=1, n_heads=4,
    )
    fm = torch.randn(2, 201, 64, requires_grad=False)
    out = conn(fm)
    out.sum().backward()
    grads = {n: p.grad for n, p in conn.named_parameters()}
    # At least the queries and the projection got gradient.
    assert grads["queries"] is not None
    assert any(
        n.startswith("proj.") and g is not None for n, g in grads.items()
    )


def test_qformer_rejects_wrong_fm_dim():
    conn = FM2Connector(
        fm_dim=64, llm_dim=128, n_query=4, n_layers=1, n_heads=4,
    )
    fm_bad = torch.randn(1, 201, 32)
    with pytest.raises(ValueError):
        conn(fm_bad)


def test_qformer_param_count_reasonable():
    conn = FM2Connector(
        fm_dim=320, llm_dim=3584, n_query=32, n_layers=2, n_heads=8,
    )
    n = conn.num_parameters()
    # Just guard that the size is in a believable range; the exact
    # number depends on implementation details.
    assert 1_000_000 < n < 50_000_000


# ---------------------------------------------------------------------------
# Templated annotations
# ---------------------------------------------------------------------------


def test_annotation_is_deterministic():
    a = annotate_specimen(
        specimen_id=42, n_atoms=11, motif="triangular_disk",
        temperature=0.4, positions=None,
    )
    b = annotate_specimen(
        specimen_id=42, n_atoms=11, motif="triangular_disk",
        temperature=0.4, positions=None,
    )
    assert a.text == b.text
    assert a.phase == b.phase


def test_annotation_mentions_required_facts():
    a = annotate_specimen(
        specimen_id=42, n_atoms=11, motif="triangular_disk",
        temperature=0.4, positions=None,
    )
    assert "11" in a.text
    assert "triangular disk" in a.text
    assert "0.40" in a.text
    assert isinstance(a, SpecimenAnnotation)


def test_annotation_phase_thresholds():
    cold = annotate_specimen(
        specimen_id=0, n_atoms=7, motif="ring", temperature=0.10, positions=None,
    )
    warm = annotate_specimen(
        specimen_id=0, n_atoms=7, motif="ring", temperature=0.50, positions=None,
    )
    hot = annotate_specimen(
        specimen_id=0, n_atoms=7, motif="ring", temperature=1.50, positions=None,
    )
    assert cold.phase == "solid-like"
    assert warm.phase == "liquid-like"
    assert hot.phase == "gas-like"


def test_annotation_with_positions_includes_geometry():
    positions = np.array(
        [[0.0, 0.0], [1.13, 0.0], [0.0, 1.13]], dtype=np.float32,
    )
    a = annotate_specimen(
        specimen_id=0, n_atoms=3, motif="triangular_disk",
        temperature=0.2, positions=positions,
    )
    assert "diameter" in a.text.lower()
    assert "coordination" in a.text.lower()
    assert a.diameter_lj is not None and a.diameter_lj > 0
    assert a.mean_coordination is not None and a.mean_coordination >= 0


def test_annotation_without_positions_drops_geometry():
    a = annotate_specimen(
        specimen_id=0, n_atoms=3, motif="ring",
        temperature=0.2, positions=None,
    )
    assert "diameter" not in a.text.lower()
    assert "coordination" not in a.text.lower()
    assert a.diameter_lj is None
    assert a.mean_coordination is None


def test_annotation_label_dict_round_trip():
    a = annotate_specimen(
        specimen_id=42, n_atoms=11, motif="triangular_disk",
        temperature=0.2, positions=None,
    )
    d = annotation_label_dict(a)
    assert d["n_atoms"] == 11
    assert d["motif"] == "triangular_disk"
    # T=0.2 is below the 0.30 solid-like upper bound (see _T_PHASES);
    # T in [0.30, 1.00) is liquid-like, T >= 1.00 is gas-like.
    assert d["phase"] == "solid-like"
    assert d["diameter_lj"] is None
    assert d["mean_coordination"] is None
