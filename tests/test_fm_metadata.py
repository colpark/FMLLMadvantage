"""Tests for the per-FM metadata schema and the shipped metadata.yaml files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fmllm.fms._schemas import FMMetadata, load_fm_metadata
from fmllm.fms._schemas.metadata_schema import (
    ConstraintDeclaration,
    DependencyDeclaration,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FM_NAMES = ("fm1_image", "fm2_rdf", "fm3_traj")


@pytest.mark.parametrize("fm_name", FM_NAMES)
def test_metadata_yaml_parses_against_schema(fm_name):
    path = REPO_ROOT / "src" / "fmllm" / "fms" / fm_name / "metadata.yaml"
    metadata = load_fm_metadata(path)
    assert metadata.name == fm_name
    assert metadata.modality
    assert metadata.input_schema.shape
    assert metadata.output_schema.semantic_name
    assert metadata.physics_constraints, "metadata declares at least one constraint"
    assert metadata.dependencies, "metadata declares at least one dependency"


@pytest.mark.parametrize("fm_name", FM_NAMES)
def test_metadata_constraint_probes_resolvable(fm_name):
    """Every declared probe path matches an importable module."""
    import importlib

    path = REPO_ROOT / "src" / "fmllm" / "fms" / fm_name / "metadata.yaml"
    metadata = load_fm_metadata(path)
    for declaration in metadata.physics_constraints:
        module = importlib.import_module(declaration.probe)
        assert hasattr(module, "run_probe")


def test_constraint_declaration_rejects_invalid_satisfaction():
    with pytest.raises(ValidationError):
        ConstraintDeclaration(
            name="x", type="hard", description="d",
            expected_satisfaction=1.5, probe="m",
        )


def test_dependency_relationship_must_be_known():
    with pytest.raises(ValidationError):
        DependencyDeclaration(
            target_variable="x", relationship="invents", confidence=0.5,
        )


def test_metadata_load_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fm_metadata(tmp_path / "absent.yaml")


def test_metadata_round_trip(tmp_path):
    """A metadata file written from a model loads back identical."""
    path = REPO_ROOT / "src" / "fmllm" / "fms" / "fm1_image" / "metadata.yaml"
    metadata = load_fm_metadata(path)
    out = tmp_path / "round.yaml"
    with out.open("w") as f:
        yaml.safe_dump(metadata.model_dump(), f, sort_keys=False)
    loaded = load_fm_metadata(out)
    assert loaded.model_dump() == metadata.model_dump()
