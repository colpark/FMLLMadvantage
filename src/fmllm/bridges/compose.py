"""Loader that builds an :class:`FMContext` from disk artifacts.

Each FM training run writes three artifacts a bridge needs:

    - ``metadata.yaml`` - declarative constraints and dependencies. This
      ships with the FM source (under ``src/fmllm/fms/<fm>/``).
    - ``probe_report.yaml`` - behavioral probe satisfaction scores from
      after training. Lives in the checkpoint directory.
    - ``calibration.json`` - split-conformal thresholds. Lives in the
      checkpoint directory.

This module locates them and wraps them into an :class:`FMContext`.
``probe_report`` and ``calibration`` are optional: a bridge can run
with degraded context (empty probe report, missing calibration) so
the orchestrator can still get bridged outputs before calibration
lands.

Produces:
    Validated :class:`FMContext` instances ready for either bridge
    flavor.

Depends on:
    pyyaml, json (stdlib).
"""

from __future__ import annotations

from pathlib import Path

from fmllm.bridges.base import FMContext
from fmllm.fms._schemas import (
    ProbeReport,
    load_fm_metadata,
    load_probe_report,
)
from fmllm.fms._schemas.probe_schema import now_utc_iso
from fmllm.fms.common import read_conformal_calibration


def metadata_yaml_path(fm_name: str) -> Path:
    """Return the path to ``metadata.yaml`` for a given FM."""
    return Path(__file__).resolve().parents[1] / "fms" / fm_name / "metadata.yaml"


def load_fm_context(
    *,
    fm_name: str,
    checkpoint_dir: Path | str | None = None,
    probe_report_path: Path | str | None = None,
    calibration_path: Path | str | None = None,
) -> FMContext:
    """Load metadata + probe report + calibration into an :class:`FMContext`.

    Either pass ``checkpoint_dir`` and the function discovers
    ``probe_report.yaml`` plus ``calibration.json`` next to the
    checkpoint, or pass the two artifact paths explicitly. Missing
    probe report or calibration falls back to empty defaults so the
    bridge can still run.
    """
    metadata = load_fm_metadata(metadata_yaml_path(fm_name))

    if probe_report_path is None and checkpoint_dir is not None:
        probe_report_path = Path(checkpoint_dir) / "probe_report.yaml"
    if calibration_path is None and checkpoint_dir is not None:
        calibration_path = Path(checkpoint_dir) / "calibration.json"

    if probe_report_path is not None and Path(probe_report_path).exists():
        probe_report = load_probe_report(probe_report_path)
    else:
        probe_report = ProbeReport(
            fm_name=metadata.name,
            fm_version=metadata.version,
            timestamp_utc=now_utc_iso(),
            results=[],
        )

    if calibration_path is not None and Path(calibration_path).exists():
        calibration = read_conformal_calibration(calibration_path)
    else:
        calibration = {}

    return FMContext(
        fm_name=metadata.name,
        metadata=metadata,
        probe_report=probe_report,
        calibration=calibration,
    )


__all__ = ["load_fm_context", "metadata_yaml_path"]
