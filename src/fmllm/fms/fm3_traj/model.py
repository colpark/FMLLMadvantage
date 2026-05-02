"""FM3: a trajectory transformer over the 100-step MD snippet.

The model takes positions and velocities for ``T`` time frames and
predicts the per-atom kinetic-energy distribution as the moments of a
Gamma distribution (shape ``alpha``, scale ``beta``).

Architecture:
    Each ``(atom, time)`` state ``(x, y, vx, vy)`` passes through a
    small atom MLP. The per-time atom embeddings aggregate via masked
    mean and masked max pooling for permutation invariance over atoms.
    A learned temporal positional embedding plus a CLS token feed a
    Transformer encoder. The CLS output projects to ``(log alpha,
    log beta)``, then ``softplus`` enforces positivity.

Permutation invariance:
    The per-time aggregation pools over real atoms with mean and max,
    both of which commute with any permutation. Padded slots get
    masked out before pooling.

Equipartition prior:
    Equipartition states ``E[KE per atom] = (d / 2) * T`` with ``d = 2``,
    so ``alpha * beta`` should match the empirical mean kinetic
    energy. The training loss adds a soft equipartition penalty that
    pulls ``alpha * beta`` toward the observed mean.

Produces:
    A dict with keys ``alpha`` and ``beta``, each shape ``(B,)``.

Depends on:
    torch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class FM3TrajTransformer(nn.Module):
    """Trajectory Transformer with permutation-invariant atom pooling."""

    def __init__(
        self,
        *,
        n_steps_input: int = 100,
        max_n_atoms: int = 30,
        embed_dim: int = 256,
        depth: int = 10,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_steps_input = n_steps_input
        self.max_n_atoms = max_n_atoms
        self.embed_dim = embed_dim

        # Atom-state encoder. Inputs: (x, y, vx, vy). 4 features per atom.
        self.atom_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        # Project the concatenated (mean, max) per-time pool back to embed_dim.
        self.time_proj = nn.Linear(2 * embed_dim, embed_dim)

        # Sequence has CLS plus up to n_steps_input + 1 frames.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, n_steps_input + 2, embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth,
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),  # (raw_alpha, raw_beta) -> softplus
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

    def forward(
        self,
        traj_positions: Tensor,
        traj_velocities: Tensor,
        atom_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            traj_positions: ``(B, T, max_n_atoms, 2)``.
            traj_velocities: ``(B, T, max_n_atoms, 2)``.
            atom_mask: ``(B, max_n_atoms)`` boolean mask flagging real atoms.

        Returns:
            ``{"alpha": (B,), "beta": (B,)}``.
        """
        if traj_positions.shape != traj_velocities.shape:
            raise ValueError(
                "traj_positions and traj_velocities must share shape; got "
                f"{tuple(traj_positions.shape)} vs {tuple(traj_velocities.shape)}"
            )
        b, t, n_max, _ = traj_positions.shape
        if t > self.n_steps_input + 1:
            raise ValueError(
                f"FM3 supports up to {self.n_steps_input + 1} frames, got {t}"
            )
        if n_max > self.max_n_atoms:
            raise ValueError(
                f"FM3 supports up to {self.max_n_atoms} atoms, got {n_max}"
            )
        if atom_mask.shape != (b, n_max):
            raise ValueError(
                f"atom_mask shape {tuple(atom_mask.shape)} does not match (B, max_n_atoms)"
            )

        atom_state = torch.cat([traj_positions, traj_velocities], dim=-1)
        atom_emb = self.atom_encoder(atom_state)  # (B, T, N_max, D)

        mask_4d = atom_mask.view(b, 1, n_max, 1).expand(-1, t, -1, atom_emb.shape[-1])
        atom_emb_zero = atom_emb.masked_fill(~mask_4d, 0.0)
        n_real = atom_mask.sum(dim=-1).clamp(min=1).to(atom_emb.dtype)
        n_real = n_real.view(b, 1, 1)
        mean_emb = atom_emb_zero.sum(dim=2) / n_real  # (B, T, D)

        very_neg = torch.finfo(atom_emb.dtype).min
        atom_emb_neg = atom_emb.masked_fill(~mask_4d, very_neg)
        max_emb = atom_emb_neg.max(dim=2)[0]  # (B, T, D)
        # Replace the rare all-padded case (no real atoms anywhere) with zeros.
        max_emb = torch.where(
            torch.isinf(max_emb) | (max_emb == very_neg),
            torch.zeros_like(max_emb),
            max_emb,
        )

        time_features = self.time_proj(torch.cat([mean_emb, max_emb], dim=-1))

        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, time_features], dim=1)
        seq = seq + self.time_pos_embed[:, : seq.shape[1]]

        seq = self.temporal_encoder(seq)
        seq = self.encoder_norm(seq)

        head_in = seq[:, 0]
        raw = self.head(head_in)
        alpha = nn.functional.softplus(raw[:, 0]) + 1.0e-3
        beta = nn.functional.softplus(raw[:, 1]) + 1.0e-3
        return {"alpha": alpha, "beta": beta}


def build_fm3_model(cfg: Any) -> FM3TrajTransformer:
    """Construct an :class:`FM3TrajTransformer` from an FM3Config-shaped object."""
    return FM3TrajTransformer(
        n_steps_input=cfg.n_steps_input,
        max_n_atoms=cfg.max_n_atoms,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    )


__all__ = ["FM3TrajTransformer", "build_fm3_model"]
