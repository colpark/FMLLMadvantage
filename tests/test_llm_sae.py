"""Tests for Phase 15 LLM-side SAE primitives.

CPU-only. Cover:

  - ActivationHarvester captures the right tensor and shape under both
    "tuple output" and "tensor output" hook regimes (Llama-style and
    bare-tensor models).
  - ActivationHarvester removes its hook on context exit and on
    explicit stop().
  - ActivationSteerer adds the right delta when output is a tuple and
    when output is a bare tensor; the coefficient and position_mask
    are honored.
  - resolve_layer_module walks dotted paths including numeric indices.
"""

from __future__ import annotations

import torch
from torch import nn

from fmllm.representation.llm_sae import (
    ActivationHarvester,
    ActivationSteerer,
    resolve_layer_module,
)


# ---------------------------------------------------------------------------
# Stub models that mimic the two "layer output" regimes.
# ---------------------------------------------------------------------------


class TupleOutLayer(nn.Module):
    """Returns ``(hidden, attn_dummy)`` like a Llama / Qwen DecoderLayer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        return (self.proj(x), torch.zeros(0))


class TensorOutLayer(nn.Module):
    """Returns just the hidden state."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class StubModel(nn.Module):
    """Two layers in a ModuleList, mimicking ``model.layers``."""

    def __init__(self, dim: int = 8, layer_cls: type = TupleOutLayer) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, dim, bias=False)
        self.layers = nn.ModuleList([layer_cls(dim), layer_cls(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


# ---------------------------------------------------------------------------
# ActivationHarvester
# ---------------------------------------------------------------------------


def test_harvester_captures_3d_residual_as_2d_flat():
    torch.manual_seed(0)
    dim = 4
    model = StubModel(dim=dim, layer_cls=TupleOutLayer)
    x = torch.randn(2, 3, dim)              # (B, T, H)
    with ActivationHarvester(model.layers[1]) as harv:
        _ = model(x)
    flat = harv.pop()
    assert flat.shape == (2 * 3, dim)
    assert flat.dtype == torch.float32


def test_harvester_handles_tensor_output_layer():
    torch.manual_seed(1)
    dim = 4
    model = StubModel(dim=dim, layer_cls=TensorOutLayer)
    x = torch.randn(1, 5, dim)
    with ActivationHarvester(model.layers[0]) as harv:
        _ = model(x)
    flat = harv.pop()
    assert flat.shape == (5, dim)


def test_harvester_pop_clears_buffer():
    dim = 4
    model = StubModel(dim=dim)
    x = torch.randn(1, 2, dim)
    harv = ActivationHarvester(model.layers[0])
    harv.start()
    _ = model(x)
    a = harv.pop()
    assert a.shape == (2, dim)
    # Subsequent pop with no forward returns empty.
    b = harv.pop()
    assert b.shape == (0,)
    harv.stop()


def test_harvester_context_exit_removes_hook():
    dim = 4
    model = StubModel(dim=dim)
    x = torch.randn(1, 1, dim)
    with ActivationHarvester(model.layers[0]) as harv:
        _ = model(x)
    # After context, another forward must not produce buffered acts.
    _ = model(x)
    flat = harv.pop()
    assert flat.shape == (1, dim)            # exactly one forward captured


def test_harvester_buffer_size_property():
    dim = 4
    model = StubModel(dim=dim)
    harv = ActivationHarvester(model.layers[0])
    assert harv.buffer_size == 0
    with harv:
        _ = model(torch.randn(2, 3, dim))
        assert harv.buffer_size == 6
        _ = model(torch.randn(1, 4, dim))
        assert harv.buffer_size == 6 + 4
    harv.pop()
    assert harv.buffer_size == 0


# ---------------------------------------------------------------------------
# ActivationSteerer
# ---------------------------------------------------------------------------


def test_steerer_adds_constant_direction_to_tuple_output():
    torch.manual_seed(0)
    dim = 4
    model = StubModel(dim=dim, layer_cls=TupleOutLayer)
    x = torch.randn(1, 3, dim)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])

    out_no_steer = model(x).clone()
    with ActivationSteerer(
        layer_module=model.layers[1],
        feature_direction=direction,
        coefficient=2.0,
    ):
        out_steered = model(x)

    # The steerer is on the LAST layer; after applying it the output
    # of the model IS the residual coming out of layer[1] plus 2*e0.
    delta = out_steered - out_no_steer
    expected = torch.zeros_like(out_steered)
    expected[..., 0] = 2.0
    assert torch.allclose(delta, expected, atol=1.0e-6)


def test_steerer_handles_tensor_output_layer():
    torch.manual_seed(0)
    dim = 4
    model = StubModel(dim=dim, layer_cls=TensorOutLayer)
    x = torch.randn(1, 2, dim)
    direction = torch.tensor([0.0, 0.0, 1.0, 0.0])

    base = model(x).clone()
    with ActivationSteerer(
        model.layers[1], feature_direction=direction, coefficient=-1.5,
    ):
        steered = model(x)
    delta = steered - base
    expected = torch.zeros_like(steered)
    expected[..., 2] = -1.5
    assert torch.allclose(delta, expected, atol=1.0e-6)


def test_steerer_position_mask_restricts_injection():
    torch.manual_seed(0)
    dim = 4
    model = StubModel(dim=dim, layer_cls=TupleOutLayer)
    x = torch.randn(1, 4, dim)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([0.0, 1.0, 1.0, 0.0])    # apply at positions 1, 2

    base = model(x).clone()
    with ActivationSteerer(
        model.layers[1], feature_direction=direction, coefficient=3.0,
        position_mask=mask,
    ):
        steered = model(x)
    delta = steered - base
    # Only positions 1 and 2 should have +3 on dim 0.
    assert torch.allclose(delta[0, 0], torch.zeros(dim), atol=1.0e-6)
    assert torch.allclose(delta[0, 3], torch.zeros(dim), atol=1.0e-6)
    assert abs(delta[0, 1, 0].item() - 3.0) < 1.0e-6
    assert abs(delta[0, 2, 0].item() - 3.0) < 1.0e-6


def test_steerer_rejects_non_1d_direction():
    dim = 4
    model = StubModel(dim=dim)
    bad_dir = torch.zeros(2, dim)
    try:
        ActivationSteerer(model.layers[0], feature_direction=bad_dir)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 2D feature_direction")


# ---------------------------------------------------------------------------
# resolve_layer_module
# ---------------------------------------------------------------------------


def test_resolve_layer_module_walks_dotted_path():
    dim = 4
    model = StubModel(dim=dim)
    layer1 = resolve_layer_module(model, "layers.1")
    assert layer1 is model.layers[1]


def test_resolve_layer_module_rejects_non_module():
    dim = 4
    model = StubModel(dim=dim)
    try:
        resolve_layer_module(model, "embed.weight")
    except TypeError:
        return
    raise AssertionError("expected TypeError when resolving to a Tensor")
