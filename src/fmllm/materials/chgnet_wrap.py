"""CHGNet wrapper -- the materials FM-equivalent for our pipeline.

Loads the public CHGNet pretrained checkpoint and exposes:

    encode(structure) -> (atom_features, pooled_embedding)

where ``atom_features`` is the per-atom feature tensor after the
final atom-graph convolution, and ``pooled_embedding`` is a
mean-pooled per-structure summary.

CHGNet uses pymatgen ``Structure`` objects (not raw arrays). To
keep the downstream pipeline aligned with our HDF5 (which stores
arrays), we provide :func:`structure_from_arrays` to build a
pymatgen Structure from the HDF5 dataset's per-specimen arrays
without any extra data dependency.

The pooled embedding is the analog of FM2's CLS embedding in the
LJ pipeline. Probes, SAEs, and the synthetic CoT all consume it.

Depends on:
    chgnet, pymatgen, torch, numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def structure_from_arrays(
    *,
    species_ids: np.ndarray,
    positions: np.ndarray,
    lattice: np.ndarray,
    element_names: list[str],
) -> Any:
    """Build a pymatgen Structure from the materials HDF5 arrays.

    Lazy-imports pymatgen so this module loads cleanly without it
    (e.g. for tests on the synthetic CoT).
    """
    from pymatgen.core import Lattice, Structure  # noqa: PLC0415

    species = [
        element_names[int(sid)] if 0 <= int(sid) < len(element_names) else "X"
        for sid in species_ids
    ]
    return Structure(
        lattice=Lattice(np.asarray(lattice, dtype=np.float64)),
        species=species,
        coords=np.asarray(positions, dtype=np.float64),
        coords_are_cartesian=True,
    )


class CHGNetWrap:
    """Wrapper around CHGNet that exposes pooled per-structure embeddings.

    Uses CHGNet's official pretrained checkpoint (small, ~5MB) as
    the FM-equivalent for materials. The wrapper:

      * loads the model on the requested device
      * registers a forward hook on the final atom-graph convolution
        layer to capture per-atom features
      * mean-pools across atoms to produce a fixed-dim embedding per
        structure
      * exposes ``predict()`` (energy / forces / stress / magmoms)
        as a passthrough for any downstream verifier-source needs

    Concrete encoder dimensions (default CHGNet config):
        atom_fea_dim = 64
        After final conv: 64-dim per atom
        Pooled per structure: 64-dim

    Usage::

        wrap = CHGNetWrap.load(device="cuda")
        atom_feas, pooled = wrap.encode(structure)   # (n_atoms, 64), (64,)
        pooled_batch = wrap.encode_batch(structures) # (B, 64)
    """

    def __init__(self, model: Any, device: str = "cuda") -> None:
        self._model = model
        self._device = device
        self._cached_atom_fea: Tensor | None = None
        self._hook_handle = None
        self._register_hook()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, *, device: str = "cuda", model_name: str = "0.3.0") -> "CHGNetWrap":
        """Download the public pretrained CHGNet and wrap it.

        Lazy-imports chgnet so the rest of the materials module is
        importable without it.
        """
        from chgnet.model import CHGNet  # noqa: PLC0415

        model = CHGNet.load(model_name=model_name)
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
        else:
            device = "cpu"
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return cls(model, device=device)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hook(self) -> None:
        """Hook the last atom-conv layer's output to capture per-atom features.

        CHGNet stores its atom-graph convolutions in
        ``self._model.atom_conv_layers`` as a ModuleList.
        """
        if self._hook_handle is not None:
            return

        def _hook(module, inputs, output):
            # output may be a tensor or tuple depending on conv impl.
            if isinstance(output, tuple):
                self._cached_atom_fea = output[0].detach()
            elif torch.is_tensor(output):
                self._cached_atom_fea = output.detach()

        layers = getattr(self._model, "atom_conv_layers", None)
        if layers is None or len(layers) == 0:
            # Some CHGNet versions name the field differently; fall
            # back to whatever final conv-shaped attribute exists.
            for name in ("atom_convs", "atom_graph_layers", "graph_convs"):
                cand = getattr(self._model, name, None)
                if cand is not None:
                    layers = cand
                    break
        if layers is None:
            raise RuntimeError(
                "Could not locate atom-graph conv layers on CHGNet. "
                "If the chgnet version changed the attribute name, "
                "patch CHGNetWrap._register_hook to point at the "
                "correct ModuleList."
            )
        self._hook_handle = layers[-1].register_forward_hook(_hook)

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, structure: Any) -> tuple[Tensor, Tensor]:
        """Encode one structure to (atom_features, pooled_embedding).

        Args:
            structure: pymatgen Structure.

        Returns:
            atom_features: (n_atoms, fea_dim) tensor on the device.
            pooled: (fea_dim,) tensor on the device.
        """
        self._cached_atom_fea = None
        with torch.no_grad():
            _ = self._model.predict_structure(structure)
        if self._cached_atom_fea is None:
            raise RuntimeError(
                "Hook did not fire; no atom features captured. Check "
                "the chgnet version vs the layer-list name in "
                "CHGNetWrap._register_hook."
            )
        atom_fea = self._cached_atom_fea
        pooled = atom_fea.mean(dim=0)
        return atom_fea, pooled

    def encode_batch(self, structures: list[Any]) -> Tensor:
        """Encode a list of structures, return (B, fea_dim) pooled embeddings.

        CHGNet's public API is one-structure-at-a-time so this loops.
        For 50K structures expect ~30-60 minutes on H100; the cached
        pooled embeddings are saved by stage 4 to amortize re-runs.
        """
        out = []
        for s in structures:
            _, pooled = self.encode(s)
            out.append(pooled.detach().cpu().unsqueeze(0))
        if not out:
            return torch.empty(0)
        return torch.cat(out, dim=0)

    # ------------------------------------------------------------------
    # Passthrough for verifier sources later
    # ------------------------------------------------------------------

    def predict(self, structure: Any) -> dict[str, Any]:
        """Run CHGNet's full prediction (energy, forces, stress, magmoms).

        Used later by the cross-FM verifier source.
        """
        with torch.no_grad():
            return self._model.predict_structure(structure)

    @property
    def device(self) -> str:
        return self._device


__all__ = ["CHGNetWrap", "structure_from_arrays"]
