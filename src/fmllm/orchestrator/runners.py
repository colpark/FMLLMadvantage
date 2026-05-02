"""FM runners: read a specimen, run the model, return the raw output dict.

The loop calls ``runner(arguments)`` whenever the LLM emits a
``call_fm`` action. Each runner knows how to translate
``arguments['specimen_id']`` (or any other supported argument) into
the right input tensors for its FM model and how to package the
forward pass into a dict the bridge consumes.

Three concrete runners ship: :class:`FM1Runner`, :class:`FM2Runner`,
:class:`FM3Runner`. The factory :func:`build_runners_from_checkpoints`
loads each FM's model + bridge once given a checkpoint root and a
training scale.

The runners hold a reference to the dataset they read from. The
caller is responsible for opening / closing the dataset.

Depends on:
    torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from fmllm.bridges import (
    StructurePreservingBridge,
    load_fm_context,
    make_structure_bridge,
)
from fmllm.data.dataset import LJSpecimenDataset
from fmllm.fms._schemas import BridgedFMOutput


FM_TO_DIR = {
    "fm1": "fm1_image",
    "fm2": "fm2_rdf",
    "fm3": "fm3_traj",
}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class FMRunner(ABC):
    """Abstract runner that calls one FM model on a dataset specimen."""

    fm_name: str

    def __init__(
        self,
        model: torch.nn.Module,
        bridge: StructurePreservingBridge,
        dataset: LJSpecimenDataset,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.eval()
        self.bridge = bridge
        self.dataset = dataset
        self.device = torch.device(device) if isinstance(device, str) else device
        self._id_to_index: dict[int, int] | None = None

    def specimen_id_to_index(self, specimen_id: int) -> int:
        """Map a specimen ID (HDF5 row in the original dataset) to the
        local dataset index. Falls back to assuming identity when the
        dataset exposes the full HDF5 file."""
        if self._id_to_index is None:
            self._id_to_index = {
                int(self.dataset[i]["specimen_id"]): i  # type: ignore[index]
                for i in range(len(self.dataset))
            }
        if specimen_id not in self._id_to_index:
            raise KeyError(f"specimen_id {specimen_id} not in dataset")
        return self._id_to_index[specimen_id]

    def __call__(self, arguments: dict[str, Any]) -> BridgedFMOutput:
        if "specimen_id" not in arguments:
            raise ValueError(
                f"{self.fm_name} runner requires 'specimen_id' in arguments"
            )
        specimen_id = int(arguments["specimen_id"])
        idx = self.specimen_id_to_index(specimen_id)
        item = self.dataset[idx]
        raw = self._forward(item)
        provenance = {"specimen_id": specimen_id, "fm_name": self.fm_name}
        return self.bridge.emit(raw, input_provenance=provenance)

    # subclass hook
    @abstractmethod
    def _forward(self, item: dict[str, Any]) -> dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Concrete runners
# ---------------------------------------------------------------------------


class FM1Runner(FMRunner):
    fm_name = "fm1"

    def _forward(self, item: dict[str, Any]) -> dict[str, Any]:
        image = item["image"]
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        image = image.to(self.device).unsqueeze(0)  # (1, H, W)
        with torch.no_grad():
            out = self.model(image)
        return {
            "count_logits": out["count_logits"][0].cpu(),
            "positions": out["positions"][0].cpu(),
            "confidence_logits": out["confidence_logits"][0].cpu(),
        }


class FM2Runner(FMRunner):
    fm_name = "fm2"

    def _forward(self, item: dict[str, Any]) -> dict[str, Any]:
        rdf = item["rdf"]
        if not isinstance(rdf, torch.Tensor):
            rdf = torch.as_tensor(rdf)
        rdf = rdf.to(self.device).unsqueeze(0)
        with torch.no_grad():
            energy = self.model(rdf)
        return {"energy": energy[0].cpu()}


class FM3Runner(FMRunner):
    fm_name = "fm3"

    def _forward(self, item: dict[str, Any]) -> dict[str, Any]:
        traj_pos = item["traj_positions"]
        traj_vel = item["traj_velocities"]
        atom_mask = item["atom_mask"]
        if not isinstance(traj_pos, torch.Tensor):
            traj_pos = torch.as_tensor(traj_pos)
        if not isinstance(traj_vel, torch.Tensor):
            traj_vel = torch.as_tensor(traj_vel)
        if not isinstance(atom_mask, torch.Tensor):
            atom_mask = torch.as_tensor(atom_mask)
        traj_pos = traj_pos.to(self.device).unsqueeze(0)
        traj_vel = traj_vel.to(self.device).unsqueeze(0)
        atom_mask = atom_mask.to(self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(traj_pos, traj_vel, atom_mask)
        return {
            "alpha": out["alpha"][0].cpu(),
            "beta": out["beta"][0].cpu(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _latest_checkpoint_dir(
    checkpoint_root: Path, fm_name: str, train_split: str,
) -> Path:
    """Find the latest run-id under ``checkpoints/<fm_dir>/<train_split>/``."""
    fm_dir = FM_TO_DIR[fm_name]
    candidates = sorted(
        (checkpoint_root / fm_dir / train_split).glob("*"),
        key=lambda p: p.name,
    )
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint under {checkpoint_root}/{fm_dir}/{train_split}/"
        )
    return candidates[-1]


def build_runners_from_checkpoints(
    *,
    checkpoint_root: Path | str,
    train_split: str,
    dataset: LJSpecimenDataset,
    cfg: Any,
    device: torch.device | str = "cpu",
) -> dict[str, FMRunner]:
    """Load each FM model from disk and wrap it in the appropriate runner.

    Args:
        checkpoint_root: Root containing ``<fm_dir>/<scale>/<run_id>/``.
        train_split: Training scale label, e.g. ``"train_50k"``.
        dataset: Open :class:`LJSpecimenDataset` the runners read from.
        cfg: A :class:`fmllm.utils.config.Config` instance providing
            architecture hyperparameters per FM.
        device: Compute device.

    Returns:
        ``{"fm1": FM1Runner, "fm2": FM2Runner, "fm3": FM3Runner}``.
    """
    from fmllm.fms.common import load_checkpoint  # local import (heavy)
    from fmllm.fms.fm1_image.model import build_fm1_model
    from fmllm.fms.fm2_rdf.model import build_fm2_model
    from fmllm.fms.fm3_traj.model import build_fm3_model

    checkpoint_root = Path(checkpoint_root)
    builders: dict[str, Callable[[Any], torch.nn.Module]] = {
        "fm1": build_fm1_model,
        "fm2": build_fm2_model,
        "fm3": build_fm3_model,
    }
    fm_cfgs = {"fm1": cfg.fm1, "fm2": cfg.fm2, "fm3": cfg.fm3}
    runner_classes: dict[str, type[FMRunner]] = {
        "fm1": FM1Runner, "fm2": FM2Runner, "fm3": FM3Runner,
    }

    runners: dict[str, FMRunner] = {}
    for short_name, fm_cfg in fm_cfgs.items():
        ckpt_dir = _latest_checkpoint_dir(checkpoint_root, short_name, train_split)
        model = builders[short_name](fm_cfg).to(device)
        load_checkpoint(ckpt_dir / "model.pt", model=model, map_location=device)
        ctx = load_fm_context(
            fm_name=FM_TO_DIR[short_name],
            checkpoint_dir=ckpt_dir,
        )
        bridge = make_structure_bridge(ctx)
        runner_cls = runner_classes[short_name]
        runners[short_name] = runner_cls(
            model=model, bridge=bridge, dataset=dataset, device=device,
        )
    return runners


__all__ = [
    "FM1Runner",
    "FM2Runner",
    "FM3Runner",
    "FMRunner",
    "build_runners_from_checkpoints",
]
