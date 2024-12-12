from typing import Tuple
import math

import einops
import torch
from torch import nn

from .functional.galerkin import galerkin_attention
from ..utility import grid


class GalerkinSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # These are not packed because of the initialization.
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self._linear_init(self.wq.weight.data)
        self._linear_init(self.wk.weight.data)
        self._linear_init(self.wv.weight.data)

        # output operator
        self.wo = nn.Linear(embed_dim, embed_dim)

        self.ln_key = nn.LayerNorm(self.head_dim)
        self.ln_value = nn.LayerNorm(self.head_dim)

    def _linear_init(self, data):
        r"""
        This is the diagonal domaination weight initialization for Q, K, V weights.
        This is described in section 5 of Galerkin Transormer, Shuhao Cao.
        The gain and delta are based on their ablation in the appendix.
        """
        with torch.no_grad():
            spread = math.sqrt(3 / self.embed_dim)
            data.uniform_(-spread, spread)
            gain, delta = 1e-2, 1e-2
            data *= gain
            diagonal_view = torch.diagonal(data)
            diagonal_view += delta

    def forward(self, seq):
        r"""
        Args:
          x: [batch, seq, embed]
        Returns:
          ga: [batch, seq, embed]
        """
        rearrange_str = "b s (heads dim) -> b heads s dim"
        query = einops.rearrange(self.wq(seq), rearrange_str, heads=self.num_heads)
        key = einops.rearrange(self.wk(seq), rearrange_str, heads=self.num_heads)
        value = einops.rearrange(self.wv(seq), rearrange_str, heads=self.num_heads)

        key = self.ln_key(key)
        value = self.ln_value(value)
        ga_heads = galerkin_attention(query, key, value)
        ga = einops.rearrange(ga_heads, "b heads s dim -> b s (heads dim)")
        ga = self.wo(ga)
        return ga


class GalerkinSubdomainSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ga = GalerkinSelfAttention(embed_dim + 2, num_heads)

    def forward(self, seq):
        batch = seq.size(0)
        seq_len = seq.size(1)
        channels = seq.size(2)

        height = seq.size(-2)
        width = seq.size(-1)

        # add point-wise positions, since always on subdomain can use (1, 1)
        g = grid((height, width), (1, 1)).to(seq.device)
        g = einops.repeat(g, "... -> b s ...", b=seq.size(0), s=seq_len)
        seq = torch.cat((g, seq), dim=1)

        seq = einops.rearrange(seq, "b s c h w -> (b s) (h w) c")
        seq = self.ga(seq)
        seq = einops.rearrange(
            seq,
            "(b s) (h w) c -> b s c h w",
            b=batch,
            s=seq_len,
            h=height,
            w=width,
        )
        return seq
