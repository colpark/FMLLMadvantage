"""Write artifact manifests with a fixed schema.

Every script that produces data, models, or results writes a manifest
YAML alongside its output. The manifest records the script identity,
inputs, configuration, timestamp, git commit, host platform, and the
versions of tracked Python packages. Downstream tools and human
reviewers rely on this schema staying consistent across the project.

Produces:
    A YAML file at the requested output path with the following
    top-level keys:
        ``schema_version`` - integer version, currently ``1``.
        ``script`` - free-form identifier of the producer.
        ``timestamp_utc`` - ISO 8601 UTC timestamp.
        ``git`` - commit hash and dirty flag if available.
        ``platform`` - python version, system, release, machine.
        ``packages`` - version per tracked dependency.
        ``inputs`` - caller-supplied dict of input descriptors.
        ``config`` - caller-supplied dict capturing run configuration.
        ``extra`` - caller-supplied free-form dict.

Depends on:
    pyyaml. Falls back gracefully when git is unavailable.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "pydantic",
    "pyyaml",
    "loguru",
    "matplotlib",
    "scikit-learn",
    "h5py",
)


def _git_commit() -> str | None:
    """Return the current git commit, or None if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return out.decode().strip()


def _git_dirty() -> bool | None:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(out.decode().strip())


def _package_versions(names: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str]:
    """Return a mapping of package name to installed version string."""
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def write_manifest(
    output_path: Path | str,
    *,
    script: str,
    inputs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a manifest YAML alongside an artifact.

    The function creates parent directories as needed and writes a YAML
    document with the project-wide manifest schema.

    Args:
        output_path: Destination for the manifest file. Conventionally
            ``<artifact_dir>/manifest.yaml`` next to the produced
            artifact.
        script: Free-form identifier of the script or pipeline stage
            that produced the artifact.
        inputs: Mapping describing the inputs the script consumed.
        config: Mapping that captures the configuration in effect.
        extra: Free-form supplementary metadata.

    Returns:
        The path written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "script": script,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_commit(),
            "dirty": _git_dirty(),
        },
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "inputs": inputs or {},
        "config": config or {},
        "extra": extra or {},
    }

    with output_path.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return output_path
