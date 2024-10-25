import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.spectral_convolution import SpectralConv
from mondrian_lib.fdm.models.ffno.ffno import SpectralConv2d as FactorizedSpectralConv
from mondrian_lib.fdm.integral import integral_2d

class SimpleFNO(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 n_modes):
        super().__init__()
        self.conv1 = SimpleSpectralConv(in_channels, hidden_channels, n_modes)
        self.conv2 = SimpleSpectralConv(hidden_channels, out_channels, n_modes)
        self.skip1 = skip_connection(in_channels, hidden_channels)
        self.skip2 = skip_connection(hidden_channels, out_channels)

    def forward(self, x):
        x = F.gelu(self.conv1(x)) + self.skip1(x)
        x = self.conv2(x) + self.skip2(x)
        return x

class SelfAttention(nn.Module):
    r"""
    This is a simple implementation of self-attention
    that is consistent in function space.
    Every input function f : D -> R^n, is assumed to be
    defined on a common reference domain D 
    """
    def __init__(self,
                 sequence_length,
                 in_channels,
                 hidden_channels,
                 heads):
        super().__init__()
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_modes = 8 
        self.scale = 1 / math.sqrt(self.hidden_channels / heads)
        self.heads = heads

        self.Q = SimpleSpectralConv(self.in_channels,
                                    self.heads * self.hidden_channels,
                                    self.n_modes)
        self.K = SimpleSpectralConv(self.in_channels,
                                    self.heads * self.hidden_channels,
                                    self.n_modes)
        self.V = SimpleSpectralConv(self.in_channels,
                                    self.heads * self.hidden_channels,
                                    self.n_modes)

        self.U = SimpleFNO(self.heads * self.hidden_channels,
                           self.in_channels,
                           self.hidden_channels,
                           self.n_modes)

    def forward(self, v):
        r"""
        Compute self attention on the input batch.
        Args:
            v: [batch, sequence-length, in_channels, H, W]
        Returns:
            u: [batch, sequence-length, in_channels, H, W]
        """
        assert v.dim() == 5
        assert v.size(1) == self.sequence_length
        assert v.size(2) == self.in_channels

        batch_size = v.size(0)
        seq_len = v.size(1)
        channels = v.size(2)
        height = v.size(3)
        width = v.size(4)

        vr = v.reshape(-1, channels, height, width)
        embed_size = (batch_size, self.heads, seq_len, -1, height, width)
        query = self.Q(vr).reshape(embed_size)
        key = self.K(vr).reshape(embed_size)
        value = self.V(vr).reshape(embed_size)

        # [batch, heads, sequence-length, sequence-length, h, w]
        inner_products = einops.einsum(
                query, key, 'b h s1 d r c, b h s2 d r c -> b h s1 s2 r c') * self.scale 
        # [batch, heads, sequence-length, sequence-length]
        l2_inner = integral_2d(inner_products, dx=1/height, dim1=-2, dim2=-1)
        A = torch.softmax(l2_inner, dim=-1)

        u_with_heads = einops.einsum(
                A, value, 'b h s1 s2, b h s2 d r c -> b h s1 d r c')
        u = einops.rearrange(u_with_heads,
                             'b h s d r c -> b s (h d) r c')

        # project to input resolution
        u = einops.rearrange(u, 'b s d h w -> (b s) d h w')
        u = self.U(u)
        u = torch.unflatten(u, dim=0, sizes=(batch_size, seq_len))

        return u
