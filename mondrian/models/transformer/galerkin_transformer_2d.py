import math

import torch
from torch import nn
import einops

from mondrian.attention.galerkin_self_attention import (
    GalerkinSelfAttention,
)
from mondrian.grid.utility import cell_centered_unit_grid
from mondrian.models.mlp import MLP


class GalerkinEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = GalerkinSelfAttention(
            embed_dim, embed_dim, num_heads, layer_norm=True
        )
        self.mlp = MLP(embed_dim, embed_dim, embed_dim, num_layers=2)

    def forward(self, seq):
        # layer normalization is applied to K and V inside attn.
        seq = self.attn(seq) + seq
        seq = self.mlp(seq) + seq
        return seq


class GalerkinTransformer2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        num_heads,
        num_layers,
        pos_method='concat'
    ):
        assert embed_dim % num_heads == 0
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pos_method = pos_method
        if pos_method == 'concat':
            lift_in_channels = in_channels + 2
        else:
            lift_in_channels = in_channels

        self.lift = MLP(lift_in_channels, embed_dim, embed_dim)
        self.project = MLP(embed_dim, out_channels, embed_dim, 2)
        self.pos_encode = MLP(2, embed_dim, embed_dim)

        self.encoders = nn.ModuleList(
            [GalerkinEncoder(embed_dim, num_heads) for _ in range(num_layers)]
        )
        
    def forward_with_concat(self, x):
        height = x.size(2)
        width = x.size(3)
        g = 2 * cell_centered_unit_grid((height, width), device=x.device) - 1
        g = einops.repeat(g, "... -> b ...", b=x.size(0))
        x = torch.cat((g, x), dim=1)
        seq = einops.rearrange(x, "b c h w -> b (h w) c")
        seq = self.lift(seq)
        for encoder in self.encoders:
            seq = encoder(seq)
        seq = self.project(seq)
        y = einops.rearrange(seq, "b (h w) c -> b c h w", h=height, w=width)
        return y
    
    def forward_with_add(self, x):
        height = x.size(2)
        width = x.size(3)

        seq = einops.rearrange(x, "b c h w -> b (h w) c")
        seq = self.lift(seq)
        
        g = 2 * cell_centered_unit_grid(
            (height, width), device=x.device
        ) - 1
        g = einops.rearrange(g, 'd h w -> (h w) d')
        pos = self.pos_encode(g)
        
        for encoder in self.encoders:
            seq = seq + pos
            seq = encoder(seq)
        seq = self.project(seq)

        y = einops.rearrange(seq, "b (h w) c -> b c h w", h=height, w=width)

        return y

    def forward(self, x, domain_size_x, domain_size_y):
        r"""
        The input is assumed to be a 2d grid. This rearranges it so the pixels can
        be interpreted as points.
        Args:
          x: [batch, channels, height, width]
        Returns:
          y: [batch, channels, height, width]
        """
        if self.pos_method == 'concat':
            return self.forward_with_concat(x)
        return self.forward_with_add(x)