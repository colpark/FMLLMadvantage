"""Self-supervised FM2 transformer with masked-RDF reconstruction.

Architecture mirrors :class:`fmllm.fms.fm2_rdf.FM2RDFTransformer`:
same bin embedding, same CLS token, same per-position embedding,
same encoder stack and final norm. Two additions:

    1. A learnable ``mask_token`` parameter that replaces masked bin
       embeddings during the SSL forward pass.
    2. A per-bin reconstruction head ``recon_head`` (MLP from
       ``embed_dim`` to 1) that predicts the masked bin values from
       the encoder output.

The :meth:`encode` method is the contract every downstream consumer
relies on (probes, connector). It returns the unmasked
``(B, rdf_bins + 1, embed_dim)`` sequence, identical in shape to
:meth:`fmllm.fms.fm2_rdf.FM2RDFTransformer.encode`, so the probing
script and the Q-Former connector drop in unchanged.

Depends on:
    torch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class FM2SSLTransformer(nn.Module):
    """1D Transformer encoder trained with masked-RDF reconstruction.

    Args:
        rdf_bins: Number of g(r) bins per specimen.
        embed_dim: Per-token hidden dimension.
        depth: Number of transformer-encoder layers.
        num_heads: Attention heads per layer.
        mlp_ratio: MLP hidden ratio.
        dropout: Dropout in attention / MLP.
    """

    def __init__(
        self,
        *,
        rdf_bins: int = 200,
        embed_dim: int = 320,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rdf_bins = rdf_bins
        self.embed_dim = embed_dim
        self.bin_embed = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, rdf_bins + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode(self, rdf: Tensor) -> Tensor:
        """Return the unmasked hidden-state sequence (no reconstruction).

        Use this for probing and for the connector. Shape matches
        :meth:`fmllm.fms.fm2_rdf.FM2RDFTransformer.encode`:
        ``(B, rdf_bins + 1, embed_dim)`` where the first token is the
        CLS summary.
        """
        if rdf.dim() != 2 or rdf.shape[-1] != self.rdf_bins:
            raise ValueError(
                f"FM2 SSL expects (B, {self.rdf_bins}) input, "
                f"got {tuple(rdf.shape)}"
            )
        batch = rdf.shape[0]
        x = self.bin_embed(rdf.unsqueeze(-1))               # (B, bins, D)
        cls = self.cls_token.expand(batch, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                      # (B, bins+1, D)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x

    def forward(
        self,
        rdf: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Reconstruct the masked bin values.

        Args:
            rdf: ``(B, rdf_bins)`` RDF values.
            mask: ``(B, rdf_bins)`` boolean mask, ``True`` where the
                bin should be hidden from the encoder.

        Returns:
            ``(B, rdf_bins)`` reconstructed bin values. Loss should
            be computed only on positions where ``mask`` is True.
        """
        if rdf.dim() != 2 or rdf.shape[-1] != self.rdf_bins:
            raise ValueError(
                f"FM2 SSL expects (B, {self.rdf_bins}) input, "
                f"got {tuple(rdf.shape)}"
            )
        if mask.shape != rdf.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match rdf shape "
                f"{tuple(rdf.shape)}"
            )
        batch = rdf.shape[0]
        x = self.bin_embed(rdf.unsqueeze(-1))                # (B, bins, D)
        # Replace masked positions with the learned mask token.
        mask_tok = self.mask_token.expand(batch, self.rdf_bins, -1)
        x = torch.where(mask.unsqueeze(-1), mask_tok, x)
        # Prepend CLS, add positional embeddings, run encoder.
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)                       # (B, bins+1, D)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.encoder_norm(x)
        # Reconstruct only the bin tokens (drop CLS).
        bins = x[:, 1:]                                      # (B, bins, D)
        return self.recon_head(bins).squeeze(-1)             # (B, bins)


def build_fm2_ssl_model(cfg: Any) -> FM2SSLTransformer:
    """Factory that mirrors :func:`fmllm.fms.fm2_rdf.build_fm2_model`.

    Reads the same FM2 config block so the SSL and supervised
    backbones share hyperparameters. ``FM2Config`` does not currently
    expose a ``dropout`` field; we honor it when present and fall
    back to the model default (0.0) otherwise. That way the connector
    code (which expects a fixed embedding dim) drops into either
    backbone unchanged.
    """
    return FM2SSLTransformer(
        rdf_bins=int(cfg.rdf_bins),
        embed_dim=int(cfg.embed_dim),
        depth=int(cfg.depth),
        num_heads=int(cfg.num_heads),
        mlp_ratio=float(cfg.mlp_ratio),
        dropout=float(getattr(cfg, "dropout", 0.0)),
    )


__all__ = ["FM2SSLTransformer", "build_fm2_ssl_model"]
