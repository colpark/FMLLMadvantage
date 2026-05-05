"""FM2 Q-Former connector.

Takes FM2's frozen hidden-state sequence ``(B, T, fm_dim)`` and
returns ``(B, n_query, llm_dim)`` continuous tokens suitable for
prepending into the LLM's input embedding stream. The Q-Former is a
small transformer with learnable query tokens that cross-attend over
the FM's per-bin embeddings.

Architecture follows the BLIP-2 pattern:

    learnable queries  ─► self-attn ─► cross-attn (over FM tokens)
       (n_query, dim)            ▲              ▲
                                 │              │
                              repeated × n_layers
                                 │
                                 ▼
                            projection (dim → llm_dim)
                                 │
                                 ▼
                          LayerNorm (in llm_dim)

The connector is small enough (typically 5-20M params for sensible
defaults) that Stage 1 alignment training fits on a single H100 in
hours.

Depends on:
    torch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _QFormerBlock(nn.Module):
    """One Q-Former layer: self-attention on queries + cross-attention to
    encoder features + MLP."""

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        # q: (B, n_query, dim), kv: (B, T, dim)
        h = self.norm_q1(q)
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + sa
        h = self.norm_q2(q)
        ca, _ = self.cross_attn(h, kv, kv, need_weights=False)
        q = q + ca
        h = self.norm_q3(q)
        q = q + self.mlp(h)
        return q


class FM2Connector(nn.Module):
    """Q-Former + projection for FM2 (RDF) representations.

    Args:
        fm_dim: FM2 hidden dim. Default matches the project's FM2.
        llm_dim: Orchestrator LLM input embedding dim. Default matches
            Qwen 2.5 7B Instruct (3584).
        n_query: Number of learnable query tokens output per specimen.
            32 is the BLIP-2 default; for RDFs (only ~200 input tokens)
            this is plenty.
        n_layers: Number of Q-Former blocks.
        n_heads: Number of attention heads inside the Q-Former.
        mlp_ratio: MLP hidden ratio inside each block.
        dropout: Dropout inside attention and MLP.
    """

    def __init__(
        self,
        *,
        fm_dim: int = 320,
        llm_dim: int = 3584,
        n_query: int = 32,
        n_layers: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fm_dim = fm_dim
        self.llm_dim = llm_dim
        self.n_query = n_query

        self.queries = nn.Parameter(torch.zeros(1, n_query, fm_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _QFormerBlock(
                    dim=fm_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.proj = nn.Linear(fm_dim, llm_dim)
        self.proj_norm = nn.LayerNorm(llm_dim)

    def forward(self, fm_features: Tensor) -> Tensor:
        """Project FM2 features into the LLM's embedding space.

        Args:
            fm_features: ``(B, T, fm_dim)`` from
                :meth:`fmllm.fms.fm2_rdf.FM2RDFTransformer.encode`.

        Returns:
            ``(B, n_query, llm_dim)`` continuous tokens ready to prepend
            into the LLM input embeddings.
        """
        if fm_features.dim() != 3:
            raise ValueError(
                f"expected (B, T, fm_dim), got shape {tuple(fm_features.shape)}"
            )
        if fm_features.shape[-1] != self.fm_dim:
            raise ValueError(
                f"expected last-dim {self.fm_dim}, got {fm_features.shape[-1]}"
            )
        batch = fm_features.shape[0]
        q = self.queries.expand(batch, -1, -1)
        for block in self.blocks:
            q = block(q, fm_features)
        out = self.proj(q)
        out = self.proj_norm(out)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["FM2Connector"]
