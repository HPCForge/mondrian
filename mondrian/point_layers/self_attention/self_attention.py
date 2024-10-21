import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math

class SelfAttention(nn.Module):
    r"""
    A simple implementation of standard softmax attention. This is
    applied to a sequence of vectors.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 heads):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.scale = 1 / math.sqrt(self.hidden_channels / heads)
        self.heads = heads

        self.Q = nn.Linear(in_channels, in_channels * heads, bias=False)
        self.K = nn.Linear(in_channels, in_channels * heads, bias=False)
        self.V = nn.Linear(in_channels, in_channels * heads, bias=False)
        self.out = nn.Linear(in_channels * heads, hidden_channels)

    def forward(self, v):
        r"""
        Compute self attention on the input batch.
        Args:
            v: [batch, sequence-length, input-dim]
        Returns:
            u: [batch, sequence-length, output-dim]
        """
        assert v.dim() == 3
        assert v.size(2) == self.in_channels

        batch_size = v.size(0)
        seq_len = v.size(1)
        channels = v.size(2)

        embed_size = (batch_size, seq_len, self.heads, channels)
        query = self.Q(v).reshape(embed_size)
        key = self.K(v).reshape(embed_size)
        value = self.V(v).reshape(embed_size)

        inner_products = einops.einsum(
                query, key, 'b s1 h c, b s2 h c -> b s1 s2 h') * self.scale 
        A = torch.softmax(inner_products, dim=-2)

        u_with_heads = einops.einsum(
                A, value, 'b s1 s2 h, b s2 h c -> b s1 h c')
        u = einops.rearrange(u_with_heads,
                             'b s h c -> b s (h c)')
        u = self.out(u)

        return u
