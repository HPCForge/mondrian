import math

import torch
from torch import nn
import einops

from mondrian.grid.attention.galerkin_self_attention import (
    GalerkinSelfAttention,
)
from mondrian.grid.utility import grid
from .mlp import MLP


class GalerkinEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = GalerkinSelfAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, embed_dim, embed_dim, num_layers=2)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.mlp(x) + x
        return x


class GalerkinTransformer2d(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, num_heads, num_layers):
        assert embed_dim % num_heads == 0
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lift = MLP(in_channels + 2, embed_dim, embed_dim)
        self.project = MLP(embed_dim, out_channels, embed_dim, 2)

        self.encoders = nn.ModuleList(
            [GalerkinEncoder(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, domain_size_x, domain_size_y):
        r"""
        The input is assumed to be a 2d grid. This rearranges it so the pixels can
        be interpreted as points.
        Args:
          x: [batch, channels, height, width]
        Returns:
          y: [batch, channels, height, width]
        """
        height = x.size(2)
        width = x.size(3)

        # add point-wise positions
        g = grid((height, width), (domain_size_y, domain_size_x)).to(x.device)
        g = einops.repeat(g, "... -> b ...", b=x.size(0))
        x = torch.cat((g, x), dim=1)

        seq = einops.rearrange(x, "b c h w -> b (h w) c")

        seq = self.lift(seq)
        for encoder in self.encoders:
            seq = encoder(seq)
        seq = self.project(seq)

        y = einops.rearrange(seq, "b (h w) c -> b c h w", h=height, w=width)

        return y
