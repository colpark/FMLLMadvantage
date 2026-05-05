"""Tests for the SteeredLLMWrapper.

CPU-only. Verify that:

  - The wrapper loads the SAE on first chat() and resolves the
    target layer.
  - The wrapped chat() invocation triggers a forward pass with the
    steering hook attached, producing a measurable change in the
    underlying model's last residual.
  - Reading the coefficient setter at runtime updates the actual
    steering strength on the next chat().
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from fmllm.representation.sae import TopKSAE
from fmllm.representation.steered_llm import SteeredLLMWrapper


# ---------------------------------------------------------------------------
# Stub LLM that lets us probe the wrapped model.
# ---------------------------------------------------------------------------


class _StubDecoderLayer(nn.Module):
    """Returns ``(hidden, dummy)`` like a Qwen2DecoderLayer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        return (self.proj(x), torch.zeros(0))


class _StubInnerModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_StubDecoderLayer(dim) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            out = layer(h)
            h = out[0]
        return h


class _StubLM(nn.Module):
    """Mimics Qwen2ForCausalLM: top-level has ``.model`` -> inner."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.model = _StubInnerModel(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _StubLLM:
    """Mimics TransformersLLM's interface (chat, _ensure_loaded, _model)."""

    def __init__(self, dim: int = 4) -> None:
        self._model = _StubLM(dim)
        self._dim = dim
        self.last_residual: torch.Tensor | None = None

    def _ensure_loaded(self) -> None:
        return

    def chat(self, messages: list[dict[str, str]]) -> str:
        # Run a single forward to expose the side-effect of steering.
        x = torch.zeros(1, 3, self._dim)
        self.last_residual = self._model(x).clone()
        return "ok"


def _save_minimal_sae(tmp_path: Path, in_dim: int, hidden_dim: int, k: int,
                     decoder_seed: int = 0) -> Path:
    """Construct a TopKSAE, save it to a sae.pt that the wrapper can load."""
    torch.manual_seed(decoder_seed)
    sae = TopKSAE(in_dim=in_dim, hidden_dim=hidden_dim, k=k,
                  normalize_decoder=False)
    sae_dir = tmp_path / "sae"
    sae_dir.mkdir(parents=True)
    payload = {
        "state_dict": {k_: v.detach().cpu()
                       for k_, v in sae.state_dict().items()},
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "k": k,
        "cls_mean": torch.zeros(in_dim).numpy(),
        "cls_std": torch.ones(in_dim).numpy(),
    }
    torch.save(payload, sae_dir / "sae.pt")
    return sae_dir


def test_wrapper_loads_sae_and_steers_residual(tmp_path: Path) -> None:
    dim = 4
    sae_dir = _save_minimal_sae(tmp_path, in_dim=dim, hidden_dim=8, k=4)

    base_llm = _StubLLM(dim=dim)
    base_llm.chat([{"role": "user", "content": "hi"}])
    baseline = base_llm.last_residual.clone()

    wrapper = SteeredLLMWrapper(
        llm=base_llm,
        sae_dir=sae_dir,
        feature_idx=2,
        coefficient=3.0,
        layer_path="model.layers.1",     # last layer of stub
    )
    _ = wrapper.chat([{"role": "user", "content": "hi"}])
    steered = base_llm.last_residual

    # The steering injection must change the residual.
    assert not torch.allclose(baseline, steered)


def test_wrapper_rejects_out_of_range_feature_idx(tmp_path: Path) -> None:
    dim = 4
    sae_dir = _save_minimal_sae(tmp_path, in_dim=dim, hidden_dim=8, k=4)
    base_llm = _StubLLM(dim=dim)
    wrapper = SteeredLLMWrapper(
        llm=base_llm, sae_dir=sae_dir,
        feature_idx=999, coefficient=1.0,
        layer_path="model.layers.0",
    )
    with pytest.raises(IndexError):
        wrapper.chat([{"role": "user", "content": "hi"}])


def test_wrapper_runtime_coefficient_setter(tmp_path: Path) -> None:
    dim = 4
    sae_dir = _save_minimal_sae(tmp_path, in_dim=dim, hidden_dim=8, k=4)
    base_llm = _StubLLM(dim=dim)
    wrapper = SteeredLLMWrapper(
        llm=base_llm, sae_dir=sae_dir,
        feature_idx=0, coefficient=1.0,
        layer_path="model.layers.0",
    )
    _ = wrapper.chat([{"role": "user", "content": "hi"}])
    res_low = base_llm.last_residual.clone()

    wrapper.coefficient = 5.0           # crank up
    _ = wrapper.chat([{"role": "user", "content": "hi"}])
    res_high = base_llm.last_residual.clone()

    # Larger coefficient must produce a larger deviation from the
    # base layer-projection on the steered position.
    delta_low = (res_low - res_low.mean()).abs().sum()
    delta_high = (res_high - res_high.mean()).abs().sum()
    assert delta_high > delta_low
