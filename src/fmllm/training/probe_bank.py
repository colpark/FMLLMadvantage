"""Probe bank: a collection of small heads on top of a frozen FM.

Phase 11 introduces inference-time CoT-over-probes. The LLM consumes
probe outputs as structured facts and reasons over them. This module
defines the storage and evaluation primitives for the probe bank.

Each probe is a 1-layer or 2-layer MLP from FM hidden dim to either
a scalar (regression) or a class logit vector (classification). The
bank is stored as a directory:

    checkpoints/probes/<run_id>/
        probe_<name>.pt
        manifest.yaml

Each ``probe_<name>.pt`` file holds the state dict plus enough
metadata to rebuild the architecture (input dim, output dim, kind,
class names if classification, hidden width).

Use:

    bank = ProbeBank.load("checkpoints/probes/.../")
    outputs = bank.evaluate(features)   # features: (B, fm_dim)
    # outputs is a list of dicts, one per row, with one key per probe

Depends on:
    torch, pyyaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Probe head
# ---------------------------------------------------------------------------


def _build_head(
    *, in_dim: int, out_dim: int, hidden: int = 0,
) -> nn.Module:
    """Build a 1- or 2-layer MLP. ``hidden=0`` means linear."""
    if hidden <= 0:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# Single probe spec
# ---------------------------------------------------------------------------


@dataclass
class ProbeSpec:
    """One probe's identity and architecture."""

    name: str
    kind: str                       # "regression" or "classification"
    in_dim: int
    out_dim: int = 1                # 1 for regression, n_classes for cls
    hidden: int = 128
    class_names: list[str] = field(default_factory=list)
    target_min: float | None = None  # for regression: training target range
    target_max: float | None = None
    target_mean: float | None = None
    target_std: float | None = None


def _build_module(spec: ProbeSpec) -> nn.Module:
    return _build_head(
        in_dim=spec.in_dim,
        out_dim=spec.out_dim,
        hidden=spec.hidden,
    )


# ---------------------------------------------------------------------------
# Probe bank: dict[name -> (ProbeSpec, nn.Module)]
# ---------------------------------------------------------------------------


class ProbeBank:
    """Frozen collection of probes evaluated together on FM features.

    The bank is constructed by calling :meth:`add` for each probe
    during training (see :func:`scripts/train_probe_bank.py`). At
    inference time, :meth:`load` rebuilds the bank from disk and
    :meth:`evaluate` applies every probe to a batch of features.
    """

    def __init__(self) -> None:
        self.specs: dict[str, ProbeSpec] = {}
        self.modules: dict[str, nn.Module] = {}

    def add(self, spec: ProbeSpec, module: nn.Module) -> None:
        if spec.name in self.specs:
            raise ValueError(f"probe {spec.name!r} already exists")
        self.specs[spec.name] = spec
        self.modules[spec.name] = module

    def to(self, device: str | torch.device) -> "ProbeBank":
        for m in self.modules.values():
            m.to(device)
        return self

    def eval(self) -> "ProbeBank":
        for m in self.modules.values():
            m.eval()
        return self

    @torch.no_grad()
    def evaluate(self, features: Tensor) -> list[dict[str, dict[str, Any]]]:
        """Apply every probe to ``features`` of shape ``(B, fm_dim)``.

        Returns a list of length B where each element is a dict
        keyed by probe name. The inner dict has at least
        ``prediction`` and a probe-specific ``confidence`` field.
        """
        if features.dim() != 2:
            raise ValueError(
                f"expected (B, fm_dim) features, got {tuple(features.shape)}"
            )
        device = next(iter(self.modules.values())).parameters().__next__().device
        x = features.to(device)
        out: list[dict[str, dict[str, Any]]] = [{} for _ in range(x.shape[0])]
        for name, spec in self.specs.items():
            mod = self.modules[name]
            mod.eval()
            logits = mod(x)
            if spec.kind == "regression":
                preds = logits.squeeze(-1).detach().cpu().tolist()
                for i, p in enumerate(preds):
                    out[i][name] = {
                        "prediction": float(p),
                        "kind": "regression",
                        # Regression confidence: how close p sits to the
                        # training-range middle (rough heuristic). The
                        # CoT generator only consults the prediction;
                        # confidence is informational.
                        "confidence": _regression_confidence(
                            float(p), spec.target_mean, spec.target_std,
                        ),
                    }
            elif spec.kind == "classification":
                probs = torch.softmax(logits, dim=-1)
                argmax = probs.argmax(dim=-1).detach().cpu().tolist()
                conf = probs.max(dim=-1).values.detach().cpu().tolist()
                full = probs.detach().cpu().tolist()
                for i, (idx, c, full_row) in enumerate(zip(argmax, conf, full, strict=True)):
                    label = (
                        spec.class_names[idx]
                        if 0 <= idx < len(spec.class_names) else str(idx)
                    )
                    out[i][name] = {
                        "prediction": label,
                        "kind": "classification",
                        "confidence": float(c),
                        "class_probs": {
                            (spec.class_names[k] if k < len(spec.class_names) else str(k)): float(p)
                            for k, p in enumerate(full_row)
                        },
                    }
            else:
                raise ValueError(
                    f"probe {name!r} has unsupported kind {spec.kind!r}"
                )
        return out

    def save(self, out_dir: Path | str) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {"probes": {}}
        for name, spec in self.specs.items():
            ckpt_path = out_dir / f"probe_{name}.pt"
            torch.save(
                {
                    "spec": {
                        "name": spec.name,
                        "kind": spec.kind,
                        "in_dim": spec.in_dim,
                        "out_dim": spec.out_dim,
                        "hidden": spec.hidden,
                        "class_names": spec.class_names,
                        "target_min": spec.target_min,
                        "target_max": spec.target_max,
                        "target_mean": spec.target_mean,
                        "target_std": spec.target_std,
                    },
                    "state_dict": {
                        k: v.detach().cpu()
                        for k, v in self.modules[name].state_dict().items()
                    },
                },
                ckpt_path,
            )
            manifest["probes"][name] = {
                "kind": spec.kind,
                "in_dim": spec.in_dim,
                "out_dim": spec.out_dim,
                "checkpoint": str(ckpt_path.name),
            }
        with (out_dir / "manifest.yaml").open("w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        return out_dir

    @classmethod
    def load(
        cls,
        out_dir: Path | str,
        device: str | torch.device = "cpu",
    ) -> "ProbeBank":
        out_dir = Path(out_dir)
        if not (out_dir / "manifest.yaml").exists():
            raise FileNotFoundError(
                f"probe bank manifest missing at {out_dir / 'manifest.yaml'}"
            )
        with (out_dir / "manifest.yaml").open("r") as f:
            manifest = yaml.safe_load(f)
        bank = cls()
        for name, info in (manifest.get("probes") or {}).items():
            payload = torch.load(
                out_dir / info["checkpoint"],
                map_location=device,
                weights_only=False,
            )
            spec_raw = payload["spec"]
            spec = ProbeSpec(
                name=spec_raw["name"],
                kind=spec_raw["kind"],
                in_dim=int(spec_raw["in_dim"]),
                out_dim=int(spec_raw["out_dim"]),
                hidden=int(spec_raw.get("hidden", 0)),
                class_names=list(spec_raw.get("class_names") or []),
                target_min=spec_raw.get("target_min"),
                target_max=spec_raw.get("target_max"),
                target_mean=spec_raw.get("target_mean"),
                target_std=spec_raw.get("target_std"),
            )
            module = _build_module(spec)
            module.load_state_dict(payload["state_dict"])
            module.to(device).eval()
            bank.add(spec, module)
        return bank

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def names(self) -> list[str]:
        return list(self.specs.keys())


def _regression_confidence(
    prediction: float, mean: float | None, std: float | None,
) -> float:
    """Heuristic confidence: 1 minus the relative distance from the
    training-distribution mean. Pure informational signal -- the CoT
    generator does not branch on this."""
    if mean is None or std is None or std == 0:
        return 1.0
    z = abs(prediction - mean) / max(std, 1.0e-6)
    return float(1.0 / (1.0 + z))


__all__ = [
    "ProbeBank",
    "ProbeSpec",
]
