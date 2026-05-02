"""Probe runner: load metadata, dispatch probes, collect results.

The runner reads an FM's ``metadata.yaml``, imports each declared
probe module via its dotted path, calls ``run_probe`` on each, and
collects the results into a :class:`ProbeReport` for the caller. Each
training script invokes the runner at the end of training.

Produces:
    A :class:`ProbeReport` covering every constraint declared in the
    FM's metadata.

Depends on:
    importlib (stdlib).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

from fmllm.fms._schemas import (
    FMMetadata,
    ProbeReport,
    ProbeResult,
    load_fm_metadata,
)
from fmllm.fms._schemas.probe_schema import now_utc_iso


def metadata_path_for(fm_name: str) -> Path:
    """Return the path to ``metadata.yaml`` for the given FM name."""
    return Path(__file__).resolve().parent / fm_name / "metadata.yaml"


def run_all_probes(
    fm_name: str,
    *,
    model: nn.Module,
    items: list[dict[str, Any]],
    device: torch.device | None = None,
    config_overrides: dict[str, dict[str, Any]] | None = None,
) -> ProbeReport:
    """Run every probe declared in the FM's ``metadata.yaml``.

    Args:
        fm_name: Name of the FM (matches the directory name under
            ``src/fmllm/fms/``).
        model: The trained model. Probes call this in eval mode.
        items: A list of per-specimen dicts collected from the
            validation loader. Each item carries the keys the probe
            needs (``image``, ``rdf``, ``traj_positions``, etc).
        device: Optional device override. Defaults to the model's
            current device.
        config_overrides: Optional per-constraint dict of probe
            kwargs to override defaults.

    Returns:
        A populated :class:`ProbeReport`.
    """
    metadata = load_fm_metadata(metadata_path_for(fm_name))
    if device is None:
        device = next(model.parameters()).device

    results: list[ProbeResult] = []
    for declaration in metadata.physics_constraints:
        config = {
            "threshold": declaration.expected_satisfaction,
        }
        if config_overrides and declaration.name in config_overrides:
            config.update(config_overrides[declaration.name])

        try:
            probe_module = importlib.import_module(declaration.probe)
        except ImportError as exc:
            logger.warning(
                "probe import failed for {}: {}", declaration.probe, exc,
            )
            results.append(
                ProbeResult(
                    constraint_name=declaration.name,
                    satisfaction_score=0.0,
                    num_test_cases=0,
                    metric="probe_import_failed",
                    passes_threshold=False,
                    threshold=declaration.expected_satisfaction,
                    details={"error": str(exc)},
                )
            )
            continue

        run_fn = getattr(probe_module, "run_probe", None)
        if run_fn is None:
            logger.warning("probe {} lacks run_probe()", declaration.probe)
            results.append(
                ProbeResult(
                    constraint_name=declaration.name,
                    satisfaction_score=0.0,
                    num_test_cases=0,
                    metric="run_probe_missing",
                    passes_threshold=False,
                    threshold=declaration.expected_satisfaction,
                    details={},
                )
            )
            continue

        result = run_fn(model=model, items=items, device=device, config=config)
        results.append(result)

    return ProbeReport(
        fm_name=metadata.name,
        fm_version=metadata.version,
        timestamp_utc=now_utc_iso(),
        results=results,
    )


def collect_items_from_loader(
    loader: torch.utils.data.DataLoader,
    *,
    n_items: int = 64,
) -> list[dict[str, Any]]:
    """Drain up to ``n_items`` items from a DataLoader as a list of dicts."""
    items: list[dict[str, Any]] = []
    for batch in loader:
        if not isinstance(batch, dict):
            raise TypeError(
                f"probe runner expects dict batches, got {type(batch).__name__}"
            )
        keys = list(batch.keys())
        any_tensor = batch[keys[0]]
        if isinstance(any_tensor, torch.Tensor):
            bs = any_tensor.shape[0]
        else:
            bs = len(any_tensor)
        for i in range(bs):
            item: dict[str, Any] = {}
            for k, v in batch.items():
                item[k] = v[i] if isinstance(v, torch.Tensor) else v[i]
            items.append(item)
            if len(items) >= n_items:
                return items
    return items


__all__ = ["collect_items_from_loader", "metadata_path_for", "run_all_probes"]
