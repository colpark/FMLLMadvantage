"""SAE on the LLM's residual stream (Phase 15).

The Phase 13/14 SAEs operate on FM2 (the small RDF foundation model).
This module operates on the LLM (Qwen) itself: train a Top-K SAE on
the residual stream at one layer of Qwen, label features by what
they fire on, and -- in a follow-up phase -- steer Qwen at inference
by clamping a feature high or low.

This is the recipe from Templeton et al. 2024 ("Scaling
Monosemanticity") and the canonical "Golden Gate Claude" demo:

    1. forward representative inputs through the LLM with a hook on
       a chosen residual-stream layer,
    2. accumulate activations across many tokens,
    3. train a Top-K SAE that decomposes the activations into sparse
       monosemantic features,
    4. at inference, hook the same layer and add a multiple of one
       SAE feature's decoder column to the residual stream to steer
       the model toward / away from that feature.

This module supplies the hooking primitives. Training and labelling
are CLI scripts that consume harvested activations; steering is a
separate baseline runner.

Two classes:

* ``ActivationHarvester``: hooks one transformer layer's output and
  accumulates the residual-stream tensor across forward passes. Use
  as a context manager so the hook is always cleaned up.
* ``ActivationSteerer``: hooks one transformer layer's output and
  injects ``coefficient * decoder_column[feature_idx]`` into the
  residual stream. The injection is additive on the post-layer
  hidden state, mirroring Templeton et al.'s steering recipe.

Both classes accept ``layer_module`` directly so they are agnostic to
the model architecture. Pick the right ``nn.Module`` for your model
in the calling code (typically ``model.model.layers[i]`` for
Llama/Qwen-family models, where the per-layer output is a tuple
``(hidden_states, ...)`` and we extract index 0).

Depends on:
    torch.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn


def _extract_residual(out: Any) -> Tensor:
    """Pull the residual-stream tensor out of a forward-hook output.

    Llama- and Qwen-family decoder layers return a tuple whose first
    element is the residual-stream hidden state. Some models return
    just the tensor. We accept both.
    """
    if isinstance(out, tuple):
        return out[0]
    if torch.is_tensor(out):
        return out
    raise TypeError(
        f"unsupported layer output type {type(out).__name__}; expected "
        f"tuple or Tensor"
    )


def _put_residual(out: Any, new_residual: Tensor) -> Any:
    """Replace the residual-stream tensor in a forward-hook output.

    Inverse of :func:`_extract_residual`: respects whether the
    original output was a tuple or a bare Tensor.
    """
    if isinstance(out, tuple):
        return (new_residual,) + tuple(out[1:])
    if torch.is_tensor(out):
        return new_residual
    raise TypeError(
        f"unsupported layer output type {type(out).__name__}"
    )


class ActivationHarvester:
    """Forward-hook that accumulates residual-stream activations.

    Args:
        layer_module: the ``nn.Module`` to hook. For Qwen / Llama,
            this is ``model.model.layers[i]``.
        store_dtype: cast captured activations to this dtype before
            buffering. Default float32 -- bf16 input gets upcast for
            stable downstream stats.

    Usage::

        harv = ActivationHarvester(layer)
        with harv:
            for chat in batches:
                model.generate(...)            # or model(...)
        all_acts = harv.pop()                  # (sum_tokens, hidden_dim)

    The hook captures the *full* (B, T, hidden_dim) residual on every
    forward and reshapes to (B*T, hidden_dim). Callers that only want
    a specific token position should slice before forwarding (e.g.
    feed only the prompt up to the position of interest).
    """

    def __init__(
        self,
        layer_module: nn.Module,
        store_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.layer_module = layer_module
        self.store_dtype = store_dtype
        self._handle: Any = None
        self._buffer: list[Tensor] = []

    def __enter__(self) -> "ActivationHarvester":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._handle is not None:
            return

        def _hook(module: nn.Module, inputs: tuple, output: Any) -> None:
            res = _extract_residual(output)
            # Detach + move to CPU + cast to a stable dtype so we
            # never hold ref to autograd graphs nor blow up VRAM.
            flat = res.detach().to("cpu", dtype=self.store_dtype)
            if flat.dim() == 3:                     # (B, T, H)
                flat = flat.reshape(-1, flat.shape[-1])
            elif flat.dim() != 2:                   # already (N, H)
                raise ValueError(
                    f"expected 2D or 3D residual, got {tuple(flat.shape)}"
                )
            self._buffer.append(flat)

        self._handle = self.layer_module.register_forward_hook(_hook)

    def stop(self) -> None:
        if self._handle is None:
            return
        self._handle.remove()
        self._handle = None

    def pop(self) -> Tensor:
        """Return the concatenated activations and clear the buffer."""
        if not self._buffer:
            return torch.empty(0, dtype=self.store_dtype)
        out = torch.cat(self._buffer, dim=0)
        self._buffer.clear()
        return out

    @property
    def buffer_size(self) -> int:
        """Total tokens captured since last :meth:`pop`."""
        return int(sum(x.shape[0] for x in self._buffer))


class ActivationSteerer:
    """Forward-hook that injects an SAE feature direction at inference.

    The Templeton et al. recipe adds a *constant* multiple of a
    feature's decoder column to the residual stream, broadcast across
    every token position. This biases the model toward (or, with a
    negative coefficient, away from) the concept the feature
    represents.

    Args:
        layer_module: same as :class:`ActivationHarvester`.
        feature_direction: ``(hidden_dim,)`` tensor. Typically the
            ``decoder.weight[:, feature_idx]`` column of the SAE,
            already on the model's device and dtype.
        coefficient: scalar multiplier. Positive amplifies, negative
            ablates. Templeton et al. used 5-10x the feature's
            empirical activation max for "Golden Gate Claude"-style
            obsession; smaller values (~0.5-2x) are saner for
            modulating without mode collapse.
        position_mask: optional ``(T,)`` boolean tensor. When
            supplied, the steering is added only at positions where
            the mask is True. Default: applied at every position.

    Usage::

        steerer = ActivationSteerer(
            layer_module=model.model.layers[14],
            feature_direction=sae.decoder.weight[:, fid].to(model.device),
            coefficient=2.0,
        )
        with steerer:
            output = model.generate(**inputs, ...)
    """

    def __init__(
        self,
        layer_module: nn.Module,
        feature_direction: Tensor,
        coefficient: float = 1.0,
        position_mask: Tensor | None = None,
    ) -> None:
        if feature_direction.dim() != 1:
            raise ValueError(
                f"feature_direction must be 1D, got "
                f"{tuple(feature_direction.shape)}"
            )
        if position_mask is not None and position_mask.dim() != 1:
            raise ValueError(
                f"position_mask must be 1D, got {tuple(position_mask.shape)}"
            )
        self.layer_module = layer_module
        self.feature_direction = feature_direction
        self.coefficient = float(coefficient)
        self.position_mask = position_mask
        self._handle: Any = None

    def __enter__(self) -> "ActivationSteerer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._handle is not None:
            return

        def _hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
            res = _extract_residual(output)
            # res: (B, T, H). Build a (B, T, H) delta tensor.
            direction = self.feature_direction.to(
                device=res.device, dtype=res.dtype,
            )
            delta = direction.view(1, 1, -1) * self.coefficient
            if self.position_mask is not None:
                mask = self.position_mask.to(
                    device=res.device, dtype=res.dtype,
                )
                if mask.shape[0] != res.shape[1]:
                    raise ValueError(
                        f"position_mask length {mask.shape[0]} != "
                        f"sequence length {res.shape[1]}"
                    )
                delta = delta * mask.view(1, -1, 1)
            new_res = res + delta
            return _put_residual(output, new_res)

        self._handle = self.layer_module.register_forward_hook(_hook)

    def stop(self) -> None:
        if self._handle is None:
            return
        self._handle.remove()
        self._handle = None


def resolve_layer_module(
    model: nn.Module, layer_path: str,
) -> nn.Module:
    """Resolve a dotted path like ``model.layers.14`` to a submodule.

    Convenience for callers that only have the model object and a
    config string. Numeric path components are treated as ``__getitem__``
    lookups (so ``layers.14`` indexes the ``ModuleList``).
    """
    obj: Any = model
    for piece in layer_path.split("."):
        if piece.isdigit():
            obj = obj[int(piece)]
        else:
            obj = getattr(obj, piece)
    if not isinstance(obj, nn.Module):
        raise TypeError(
            f"resolved path {layer_path!r} is not an nn.Module "
            f"(got {type(obj).__name__})"
        )
    return obj


__all__ = [
    "ActivationHarvester",
    "ActivationSteerer",
    "resolve_layer_module",
]
