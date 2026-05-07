"""Top-K sparse autoencoder over FM2 CLS embeddings.

Phase 13 follows the recent (Anthropic, OpenAI) approach: a Top-K
SAE replaces the older L1-penalty SAE. The encoder is a single linear
layer; after activation, only the top ``k`` activations per row are
kept (the rest are zeroed). The decoder is another linear layer that
reconstructs the original embedding. Loss is plain MSE on
reconstruction; sparsity is enforced structurally by the Top-K, not
by an L1 term.

This is simpler to train, easier to interpret, and avoids the
``λ`` tuning headache.

Usage::

    sae = TopKSAE(in_dim=320, hidden_dim=1024, k=32)
    recon, latent = sae(cls_embedding)        # (B, 320), (B, 1024) sparse
    loss = ((recon - cls_embedding) ** 2).mean()

Depends on:
    torch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class TopKSAE(nn.Module):
    """Single-layer encoder + Top-K + single-layer decoder.

    Args:
        in_dim: Input dimension. For FM2 CLS this is 320.
        hidden_dim: Latent dimension. Typically 3-8x the input.
            1024 keeps memory modest; 2048 gives more feature
            granularity at moderate cost.
        k: Number of active latents per row. 32 is the
            project default; 16 is more aggressive sparsity, 64
            looser.
        normalize_decoder: When True, normalize decoder columns to
            unit norm after each step. Standard SAE practice that
            stabilizes training.
        bias_decoder: When True, adds a learnable bias on the
            decoder output (the "decoder bias trick" some recipes
            recommend).
    """

    def __init__(
        self,
        *,
        in_dim: int = 320,
        hidden_dim: int = 1024,
        k: int = 32,
        normalize_decoder: bool = True,
        bias_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.normalize_decoder = normalize_decoder

        self.encoder = nn.Linear(in_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, in_dim, bias=bias_decoder)

        # The "pre-encoder bias" trick: subtract a learnable mean
        # before encoding and add it back after decoding. This is a
        # widely-used SAE trick that lets the encoder focus on
        # structure rather than the dataset mean.
        self.pre_bias = nn.Parameter(torch.zeros(in_dim))

    @torch.no_grad()
    def _renormalize_decoder(self) -> None:
        """Project decoder columns onto the unit sphere. Call after
        each optimizer step."""
        if not self.normalize_decoder:
            return
        w = self.decoder.weight.data           # (in_dim, hidden_dim)
        norms = w.norm(dim=0, keepdim=True).clamp_min(1.0e-8)
        self.decoder.weight.data = w / norms

    def encode(self, x: Tensor) -> Tensor:
        """Run the encoder and apply Top-K. Returns the sparse latent."""
        x_centered = x - self.pre_bias
        z = self.encoder(x_centered)
        z = torch.nn.functional.relu(z)
        # Top-K mask: keep only the largest k activations per row.
        if self.k < self.hidden_dim:
            topk_vals, topk_idx = z.topk(self.k, dim=-1, sorted=False)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, topk_idx, 1.0)
            z = z * mask
        return z

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z) + self.pre_bias

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (reconstruction, sparse latent)."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def resample_dead_features(
        self,
        activation_history: Tensor,
        x_batch_normalized: Tensor,
        threshold: int = 0,
        optimizer: Any | None = None,
        encoder_init_scale: float = 0.2,
    ) -> int:
        """Reinitialize features that haven't fired in the recent window.

        Bricken et al. 2023 (Anthropic) recipe for keeping Top-K SAEs
        from accumulating dead features:

          1. Find features whose recent fire-count <= threshold.
          2. Sample input rows proportional to per-row reconstruction
             error (high-error rows reveal directions the SAE doesn't
             yet cover).
          3. For each dead feature, set its decoder column to a
             unit-norm copy of the sampled row's residual; reset its
             encoder row to the same direction (scaled smaller); zero
             the bias.
          4. Reset the optimizer's first/second moment for the
             reinitialized weights so the new direction isn't fought
             by stale momentum.

        Args:
            activation_history: ``(hidden_dim,)`` count of fires in the
                most recent training window.
            x_batch_normalized: ``(B, in_dim)`` batch in the
                cls-normalized space the SAE works in.
            threshold: features with <= this count are considered dead.
            optimizer: AdamW (or similar) optimizer holding the SAE
                parameters. Required to reset the moments.
            encoder_init_scale: how strongly to push the encoder
                toward the new direction (0.2 in Anthropic's reference).

        Returns:
            The number of features reinitialized.
        """
        device = self.encoder.weight.device
        history = activation_history.to(device)
        dead_idx = torch.where(history <= threshold)[0]
        n_dead = int(dead_idx.numel())
        if n_dead == 0:
            return 0
        if x_batch_normalized.numel() == 0:
            return 0

        x = x_batch_normalized.to(device)
        recon, _ = self(x)
        per_row_err = ((x - recon) ** 2).sum(dim=-1)         # (B,)
        if float(per_row_err.sum().item()) <= 0.0:
            # All rows reconstruct perfectly; nothing to bias toward.
            return 0
        probs = per_row_err / per_row_err.sum()
        sampled = torch.multinomial(probs, n_dead, replacement=True)
        new_dirs = (x[sampled] - self.pre_bias)              # (n_dead, in_dim)
        norms = new_dirs.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        new_dirs_unit = new_dirs / norms

        # Decoder is shape (in_dim, hidden_dim); column j is the
        # direction added to the residual when feature j is active.
        self.decoder.weight.data[:, dead_idx] = new_dirs_unit.T

        # Encoder is shape (hidden_dim, in_dim); row j is the
        # direction projected onto x to compute z_j.
        self.encoder.weight.data[dead_idx] = new_dirs_unit * encoder_init_scale
        self.encoder.bias.data[dead_idx] = 0.0

        # Reset Adam-style optimizer moments so the new direction is
        # not pulled back by stale momentum from the dead direction.
        if optimizer is not None:
            for group in optimizer.param_groups:
                for p in group.get("params", []):
                    state = optimizer.state.get(p, None)
                    if not state:
                        continue
                    # Encoder weight rows
                    if p is self.encoder.weight:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in state:
                                state[key][dead_idx] = 0.0
                    # Encoder bias entries
                    elif p is self.encoder.bias:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in state:
                                state[key][dead_idx] = 0.0
                    # Decoder weight columns
                    elif p is self.decoder.weight:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in state:
                                state[key][:, dead_idx] = 0.0

        # Renormalize after the in-place edit.
        self._renormalize_decoder()
        return n_dead


def build_topk_sae(
    *,
    in_dim: int = 320,
    hidden_dim: int = 1024,
    k: int = 32,
) -> TopKSAE:
    """Factory mirroring the project's ``build_*_model`` style."""
    return TopKSAE(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        k=k,
        normalize_decoder=True,
        bias_decoder=True,
    )


__all__ = ["TopKSAE", "build_topk_sae"]
