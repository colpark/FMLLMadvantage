"""FM1: a small Vision Transformer with a DETR-style set-prediction head.

The model takes a 64x64 grayscale image of a Lennard-Jones cluster and
predicts:

    1. The atom count ``N`` as a categorical distribution over
       ``{0, 1, ..., max_n_atoms}``.
    2. Up to ``num_queries`` candidate atom positions ``(x, y)`` in LJ
       units, with one confidence (objectness) logit per query slot.

Architecture:
    Patch embedding (Conv2d, kernel=stride=patch_size) -> CLS prepended
    -> learned absolute positional embedding -> Transformer encoder ->
    learned object queries cross-attend to the encoded patch sequence
    via a Transformer decoder. The CLS token feeds the count head; each
    decoded query feeds the position and confidence heads.

Translation equivariance:
    The conv-based patch embedding gives exact translation equivariance
    when the image shifts by integer multiples of ``patch_size`` pixels.
    The learned absolute positional embeddings break exact equivariance
    for non-multiples, but the patch-grid prior plus convolutional
    feature aggregation supplies a strong inductive bias that the
    Hungarian-matched position loss exploits during training. The
    DETR-style head emits absolute LJ coordinates rather than relative
    offsets, so equivariance manifests as: a translation of the input
    image by ``(dx_px, dy_px)`` shifts the predicted positions by
    ``(dx_px, dy_px) * pixel_size_lj`` in expectation.

Produces:
    A dict with keys ``count_logits``, ``positions``, and
    ``confidence_logits``.

Depends on:
    torch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class FM1ImageViT(nn.Module):
    """Vision Transformer with DETR-style set-prediction head."""

    def __init__(
        self,
        *,
        image_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_queries: int = 32,
        max_n_atoms: int = 30,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}"
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_queries = num_queries
        self.max_n_atoms = max_n_atoms

        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.count_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, max_n_atoms + 1),
        )
        self.position_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )
        self.confidence_head = nn.Linear(embed_dim, 1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_embed, std=0.02)

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            image: ``(B, H, W)`` or ``(B, 1, H, W)`` grayscale image.

        Returns:
            ``count_logits`` of shape ``(B, max_n_atoms + 1)``,
            ``positions`` of shape ``(B, num_queries, 2)`` in LJ units,
            ``confidence_logits`` of shape ``(B, num_queries)``.
        """
        if image.dim() == 3:
            image = image.unsqueeze(1)
        if image.dim() != 4 or image.shape[1] != 1:
            raise ValueError(
                f"FM1 expects (B, 1, H, W) or (B, H, W) input, got {tuple(image.shape)}"
            )
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            raise ValueError(
                f"FM1 expects {self.image_size}x{self.image_size} images, got {tuple(image.shape)}"
            )

        batch = image.shape[0]
        x = self.patch_embed(image)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)

        cls_out = x[:, 0]
        memory = x[:, 1:]

        queries = self.query_embed.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        decoded = self.decoder_norm(decoded)

        positions = self.position_head(decoded)  # (B, Q, 2)
        confidence_logits = self.confidence_head(decoded).squeeze(-1)  # (B, Q)
        count_logits = self.count_head(cls_out)
        return {
            "count_logits": count_logits,
            "positions": positions,
            "confidence_logits": confidence_logits,
        }


def build_fm1_model(cfg: Any) -> FM1ImageViT:
    """Construct an :class:`FM1ImageViT` from an FM1Config-shaped object."""
    return FM1ImageViT(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        encoder_depth=cfg.encoder_depth,
        decoder_depth=cfg.decoder_depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        num_queries=cfg.num_queries,
        max_n_atoms=cfg.max_n_atoms,
    )


__all__ = ["FM1ImageViT", "build_fm1_model"]
