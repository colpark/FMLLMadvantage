"""FM2: 1D Transformer over a length-200 radial distribution function.

The model maps the RDF ``g(r)`` of a Lennard-Jones cluster to the
coarse-grained per-atom potential energy. A CLS token aggregates the
sequence, and a small MLP head projects the CLS embedding to a scalar
energy prediction.

Permutation invariance:
    The RDF ``g(r)`` is permutation-invariant by construction: it
    depends only on the multi-set of pairwise distances. The model
    consumes the RDF directly, so any permutation of the upstream
    atom labels produces the same input and the same prediction. No
    architectural symmetry constraint needed.

Extensive scaling:
    The output is per-atom energy. Total energy = N * per_atom_energy
    by output design. The training and downstream code rely on this
    extensiveness convention.

Produces:
    A scalar per-atom energy in LJ units.

Depends on:
    torch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class FM2RDFTransformer(nn.Module):
    """1D Transformer encoder with a CLS token over the RDF sequence."""

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
        self.bin_embed = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
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

        self.energy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode(self, rdf: Tensor) -> Tensor:
        """Return the full hidden-state sequence without applying the head.

        The pre-head representation is what probes (Phase 9.0) and the
        Layer C connector (Phase 9.A) consume. ``forward`` calls into
        this and then applies the energy head; both paths share one
        backbone forward pass.

        Args:
            rdf: ``(B, rdf_bins)`` tensor with the RDF values.

        Returns:
            ``(B, rdf_bins + 1, embed_dim)`` tensor whose first token
            along axis 1 is the CLS summary the energy head reads.
        """
        if rdf.dim() != 2 or rdf.shape[-1] != self.rdf_bins:
            raise ValueError(
                f"FM2 expects (B, {self.rdf_bins}) input, got {tuple(rdf.shape)}"
            )
        batch = rdf.shape[0]
        x = self.bin_embed(rdf.unsqueeze(-1))
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x

    def forward(self, rdf: Tensor) -> Tensor:
        """Map ``g(r)`` to a scalar per-atom energy.

        Args:
            rdf: ``(B, rdf_bins)`` tensor with the RDF values.

        Returns:
            Tensor of shape ``(B,)``.
        """
        x = self.encode(rdf)
        cls_out = x[:, 0]
        return self.energy_head(cls_out).squeeze(-1)


def build_fm2_model(cfg: Any) -> FM2RDFTransformer:
    """Construct an :class:`FM2RDFTransformer` from an FM2Config-shaped object."""
    return FM2RDFTransformer(
        rdf_bins=cfg.rdf_bins,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    )


__all__ = ["FM2RDFTransformer", "build_fm2_model"]
