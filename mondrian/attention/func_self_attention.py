from typing import Tuple
import math

import einops
import torch
from torch import nn

from .functional.func_attention import func_attention
from ..spectral_conv import SimpleSpectralConv2d
from ..log_cpb import LogCPB
from ..decompose import decompose2d, recompose2d
from ..utility import is_power_of_2
from ..quadrature import Quadrature2d
from ...constants import HEAD_SPLIT_OPTIONS, CHANNEL, SPATIAL

from .galerkin_self_attention import GalerkinSubdomainSA


@torch.compile
class FuncSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_split: str,
        use_bias: bool,
        quadrature_method: str,
    ):
        super().__init__()
        assert head_split in HEAD_SPLIT_OPTIONS
        assert is_power_of_2(num_heads)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert is_power_of_2(self.head_dim)
        self.head_split = head_split
        self.use_bias = use_bias
        self.score_method = quadrature_method

        modes = 2
        self.qkv_operator = SimpleSpectralConv2d(embed_dim, 3 * embed_dim, modes)
        self.output_operator = SimpleSpectralConv2d(embed_dim, embed_dim, modes)
        # self.qkv_operator = GalerkinSubdomainSA(
        #    embed_dim, 3 * embed_dim, num_heads=1, method=quadrature_method
        # )
        # self.output_operator = GalerkinSubdomainSA(
        #    embed_dim, embed_dim, num_heads=1, method=quadrature_method
        # )

        if use_bias:
            self.log_cpb = LogCPB(embed_dim, num_heads)
        else:
            self.log_cpb = None

        self.quadrature = Quadrature2d(quadrature_method)

    def _qkv(self, v):
        dims = [1 for _ in range(v.dim())]
        dims[2] = 3
        v = self.qkv_operator(v)
        return v

    def _forward_channel_heads(self, seq, n_sub_x, n_sub_y):
        r"""
        Computes the multihead by partitioning along the channels axis
        """
        query, key, value = einops.rearrange(
            self._qkv(seq),
            "b s (qkv num_heads head_dim) ... -> qkv b num_heads s head_dim ...",
            qkv=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        if self.log_cpb is not None:
            bias = self.log_cpb(n_sub_x, n_sub_y, device=query.device)
        else:
            bias = None

        quadrature_weights = self.quadrature(query)
        sa = func_attention(
            query,
            key,
            value,
            quadrature_weights,
            bias=bias,
        )
        sa = einops.rearrange(sa, "b h s d ... -> b s (h d) ...")
        return self.output_operator(sa)

    def forward(self, seq, n_sub_x, n_sub_y):
        r"""
        Args:
          seq: [batch, seq, channels, ...], a sequence of functions
          n_sub_x: number of subdomains in the x-direction
          n_sub_y: number of subdomains in the y-direction
        Returns:
          [batch, seq, channels, ...]
        """
        return self._forward_channel_heads(seq, n_sub_x, n_sub_y)
