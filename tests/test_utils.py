"""Local tests for the utility modules.

None of these tests require a GPU. They run on a laptop after
``uv sync --extra dev``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fmllm.utils.config import Config, load_config
from fmllm.utils.logging import configure_logging
from fmllm.utils.manifests import SCHEMA_VERSION, write_manifest
from fmllm.utils.run_ids import generate_run_id


# ---------------------------------------------------------------------------
# run_ids
# ---------------------------------------------------------------------------


def test_run_id_canonical_format() -> None:
    rid = generate_run_id("fm1-train", now=datetime(2026, 3, 15, 14, 15, 22))
    assert rid == "20260315-141522-fm1-train"


def test_run_id_sanitizes_slug() -> None:
    rid = generate_run_id(
        "FM1 Train! Baseline.",
        now=datetime(2026, 3, 15, 14, 15, 22),
    )
    assert rid == "20260315-141522-fm1-train-baseline"


def test_run_id_rejects_empty_slug() -> None:
    with pytest.raises(ValueError):
        generate_run_id("!!!")


def test_run_id_uses_now_by_default() -> None:
    rid = generate_run_id("smoke")
    assert rid.endswith("-smoke")
    # The timestamp prefix has the right shape: YYYYMMDD-HHMMSS, 15 chars.
    assert len(rid.split("-smoke")[0]) == 15


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_default_config_matches_plan() -> None:
    cfg = Config()
    assert cfg.dataset.num_specimens == 50_000
    assert cfg.dataset.num_holdout == 10_000
    assert 5 in cfg.dataset.n_in_distribution
    assert 30 in cfg.dataset.n_ood
    assert cfg.fm1.name == "fm1_image"
    assert cfg.fm2.name == "fm2_rdf"
    assert cfg.fm3.name == "fm3_traj"
    assert cfg.orchestrator.llm_model.startswith("meta-llama/")


def test_load_config_round_trip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("seeds:\n  numpy: 42\n  torch: 7\n  python: 3\n")
    cfg = load_config(cfg_path)
    assert cfg.seeds.numpy == 42
    assert cfg.seeds.torch == 7
    assert cfg.seeds.python == 3


def test_load_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("not_a_real_section:\n  foo: 1\n")
    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_load_config_repo_default() -> None:
    """The shipped configs/default.yaml must validate against the schema."""
    repo_root = Path(__file__).resolve().parents[1]
    default_path = repo_root / "configs" / "default.yaml"
    assert default_path.exists(), "configs/default.yaml is missing"
    cfg = load_config(default_path)
    assert isinstance(cfg, Config)


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


def test_write_manifest_contains_required_fields(tmp_path: Path) -> None:
    out = tmp_path / "manifest.yaml"
    write_manifest(
        out,
        script="test-script",
        inputs={"a": 1},
        config={"b": 2},
        extra={"c": 3},
    )
    assert out.exists()
    with out.open() as f:
        m = yaml.safe_load(f)
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["script"] == "test-script"
    assert m["inputs"]["a"] == 1
    assert m["config"]["b"] == 2
    assert m["extra"]["c"] == 3
    assert "timestamp_utc" in m
    assert "platform" in m
    assert "packages" in m
    assert "git" in m


def test_write_manifest_defaults_to_empty_dicts(tmp_path: Path) -> None:
    out = tmp_path / "manifest.yaml"
    write_manifest(out, script="bare")
    with out.open() as f:
        m = yaml.safe_load(f)
    assert m["inputs"] == {}
    assert m["config"] == {}
    assert m["extra"] == {}


def test_write_manifest_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "manifest.yaml"
    write_manifest(out, script="bare")
    assert out.exists()


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_configure_logging_writes_file(tmp_path: Path) -> None:
    log_path = configure_logging(tmp_path / "run")
    assert log_path.parent.exists()
    assert log_path.name == "run.log"

    from loguru import logger

    logger.info("hello from test")
    logger.complete()
    assert log_path.exists()
    content = log_path.read_text()
    assert "hello from test" in content
