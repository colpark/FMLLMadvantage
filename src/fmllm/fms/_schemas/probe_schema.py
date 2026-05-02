"""Pydantic schema for behavioral probe results (Layer 3).

Each probe declared in an FM's ``metadata.yaml`` runs after training
and emits a :class:`ProbeResult`. The collection of results per FM
ships as a ``probe_report.yaml`` file alongside the trained checkpoint.
Bridges then read the satisfaction scores when they assemble bridged
outputs.

Produces:
    Validated :class:`ProbeResult` and :class:`ProbeReport` objects
    plus YAML save/load helpers.

Depends on:
    pydantic, pyyaml.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProbeResult(_StrictModel):
    """One probe's outcome."""

    constraint_name: str
    satisfaction_score: float = Field(ge=0.0, le=1.0)
    num_test_cases: int = Field(ge=0)
    metric: str
    passes_threshold: bool
    threshold: float
    details: dict[str, Any] = Field(default_factory=dict)


class ProbeReport(_StrictModel):
    """Aggregate of every probe result for one FM."""

    fm_name: str
    fm_version: str
    timestamp_utc: str
    results: list[ProbeResult] = Field(default_factory=list)

    def by_constraint(self, name: str) -> ProbeResult | None:
        for r in self.results:
            if r.constraint_name == name:
                return r
        return None


def save_probe_report(
    report: ProbeReport,
    path: Path | str,
) -> Path:
    """Write a probe report to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(report.model_dump(), f, sort_keys=False)
    return path


def load_probe_report(path: Path | str) -> ProbeReport:
    """Read a probe report from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"probe report not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return ProbeReport.model_validate(raw)


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ProbeReport",
    "ProbeResult",
    "load_probe_report",
    "now_utc_iso",
    "save_probe_report",
]
