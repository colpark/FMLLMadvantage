"""Tests for BridgedFMOutput round-trip and per-FM value payloads."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fmllm.fms._schemas import (
    ApplicableConstraint,
    BridgedDependency,
    BridgedFMOutput,
    Prediction,
    Source,
    Uncertainty,
)
from fmllm.fms.fm1_image.bridge_schema import AtomPosition, AtomSet
from fmllm.fms.fm2_rdf.bridge_schema import EnergyPerAtom
from fmllm.fms.fm3_traj.bridge_schema import GammaKEDistribution


def _sample_atom_set() -> dict:
    return AtomSet(
        n_atoms_pred=2,
        positions=[
            AtomPosition(x_lj=0.0, y_lj=0.0, confidence=0.95),
            AtomPosition(x_lj=1.122, y_lj=0.0, confidence=0.90),
        ],
        raw_count_logits=[0.0] * 31,
        raw_query_count=4,
    ).model_dump()


def test_bridge_output_round_trips_through_json():
    obj = BridgedFMOutput(
        prediction=Prediction(
            quantity="atom_positions_lj",
            value=_sample_atom_set(),
            units="lj_units",
            uncertainty=Uncertainty(lower=0.0, upper=0.15, confidence_level=0.9),
        ),
        source=Source(
            fm_name="fm1_image",
            fm_version="0.1.0",
            in_distribution=True,
            raw_input_provenance={"specimen_id": 42},
        ),
        applicable_constraints=[
            ApplicableConstraint(
                constraint_name="positions_in_box", type="hard",
                satisfied_in_training=True, satisfaction_score=0.99,
            ),
        ],
        dependencies=[
            BridgedDependency(
                target_variable="atom_count",
                relationship="derives",
                derived_value=2,
                confidence=0.95,
            ),
        ],
        timestamp="2026-05-02T01:00:00+00:00",
    )

    payload = obj.model_dump_json()
    parsed = json.loads(payload)
    rehydrated = BridgedFMOutput.model_validate(parsed)
    assert rehydrated.model_dump() == obj.model_dump()


def test_bridge_output_rejects_extra_top_level_keys():
    with pytest.raises(ValidationError):
        BridgedFMOutput(
            prediction=Prediction(
                quantity="x", value=0.0, units="lj",
            ),
            source=Source(
                fm_name="fm2_rdf", fm_version="0.1.0", in_distribution=True,
            ),
            applicable_constraints=[],
            dependencies=[],
            timestamp="2026-05-02T01:00:00+00:00",
            extra_field="not allowed",
        )


def test_uncertainty_supports_per_element_list():
    u = Uncertainty(
        lower=[0.0, 0.0, 0.0],
        upper=[0.1, 0.1, 0.1],
        confidence_level=0.9,
    )
    assert isinstance(u.lower, list)
    assert isinstance(u.upper, list)


def test_per_fm_value_payloads_validate():
    fm1_value = AtomSet(
        n_atoms_pred=0,
        positions=[],
        raw_count_logits=[0.0],
        raw_query_count=4,
    )
    assert fm1_value.n_atoms_pred == 0

    fm2_value = EnergyPerAtom(value_lj=-0.85)
    assert fm2_value.value_lj == pytest.approx(-0.85)

    fm3_value = GammaKEDistribution(
        alpha=2.0, beta=0.5, mean=1.0, variance=0.5,
        implied_temperature_lj=1.0,
    )
    assert fm3_value.implied_temperature_lj == pytest.approx(1.0)


def test_atom_set_position_confidence_in_unit_interval():
    with pytest.raises(ValidationError):
        AtomPosition(x_lj=0.0, y_lj=0.0, confidence=1.5)
