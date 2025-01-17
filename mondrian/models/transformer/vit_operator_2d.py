from typing import Union, Tuple

import einops
import torch
from torch import nn

from mondrian.attention.func_self_attention import FuncSelfAttention
from mondrian.grid.decompose import decompose2d, recompose2d
from mondrian.layers.pointwise import PointwiseMLP2d
from mondrian.layers.seq_op import seq_op
from mondrian.layers.learned_pos_embedding import LearnedPosEmbedding2d
from mondrian.grid.utility import cell_centered_grid, cell_centered_unit_grid
from mondrian.layers.feed_forward_operator import get_default_feed_forward_operator
from mondrian.layers.spectral_conv import SpectralConvNeuralOperator

class SequenceInstanceNorm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(embed_dim)

    def forward(self, v):
        return seq_op(self.norm, v)


class SequenceGroupNorm2d(nn.Module):
    def __init__(self, num_groups, embed_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, embed_dim)

    def forward(self, v):
        return seq_op(self.norm, v)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, head_split, use_bias):
        super().__init__()
        self.sa = FuncSelfAttention(
            embed_dim, num_heads, head_split, use_bias
        )
        
        self.mlp = get_default_feed_forward_operator(embed_dim, embed_dim, embed_dim)
        self.norm1 = SequenceGroupNorm2d(8, embed_dim)
        self.norm2 = SequenceGroupNorm2d(8, embed_dim)

    def forward(self, v, n_sub_x, n_sub_y):
        with torch.profiler.record_function("self_attention"):
            v = self.sa(self.norm1(v), n_sub_x, n_sub_y) + v
        with torch.profiler.record_function("mlp"):
            v = self.mlp(self.norm2(v)) + v
        return v

class ViTOperator2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        head_split: str,
        num_layers: int,
        max_seq_len: int,
        subdomain_size: Union[int, Tuple[int, int]],
    ):
        r"""
        An implementation of a ViT-style operator, specific to regular 2d grids.
        The `head_dim` is computed as `embed_dim // num_heads`
        Parameters:
          in_channels: The expected number of channels input to the model.
          out_channels: The number of channels output by the model.
          embed_dim: The number of channels used in the attention operators.
          num_heads: The number of heads used in multihead attention.
          head_split: way to split heads for multihead attention. ['spatial', 'channel']
          num_layers: The number of Encoder blocks.
          sub_domain_size: The physical subdomain size. This is independent of
                           the input discretization. It should correspond to some
                           "physical" dimension, relative to the global domain size.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(subdomain_size, int):
            subdomain_size = (subdomain_size, subdomain_size)
        assert isinstance(subdomain_size[0], int)
        assert isinstance(subdomain_size[1], int)

        self.subdomain_size = subdomain_size
        self.sub_size_y = self.subdomain_size[0]
        self.sub_size_x = self.subdomain_size[1]

        self.encoder = nn.ModuleList(
            [
                Encoder(embed_dim, num_heads, head_split, False)
                for _ in range(num_layers)
            ]
        )

        self.input_project = get_default_feed_forward_operator(in_channels, embed_dim, hidden_channels=embed_dim)
        self.output_project = get_default_feed_forward_operator(embed_dim, out_channels, hidden_channels=embed_dim)

        self.pos_embedding = LearnedPosEmbedding2d(
            seq_len=max_seq_len, channels=embed_dim
        )

    def forward(self, v: torch.Tensor, domain_size_y: int, domain_size_x: int):
        r"""
        Args:
          v: [batch_size x in_channels x H x W], the input function discretized on a
             regular grid.
          domain_size_y: The size, in the y-direction, corresponding to the axis H,
                         of domain that the function v is defined on.
          domain_size_x: The domain's size in the x-direction.
        Returns:
          u: [batch_size x out_channels x H x W]
        """
        assert v.size(1) == self.in_channels
        assert isinstance(domain_size_y, int)
        assert isinstance(domain_size_x, int)
        assert domain_size_y % self.sub_size_y == 0
        assert domain_size_x % self.sub_size_x == 0
        n_sub_y = domain_size_y // self.sub_size_y
        n_sub_x = domain_size_x // self.sub_size_x

        d = decompose2d(v, n_sub_x, n_sub_y)
        d = self.input_project(d)
        d = self.pos_embedding(d)
        
        for encoder in self.encoder:
            d = encoder(d, n_sub_x, n_sub_y)

        d = self.output_project(d)
        u = recompose2d(d, n_sub_x, n_sub_y)
        
        return u
    
class ViTOperatorFixedPosEmbedding2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        *args,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, embed_dim, *args, **kwargs)
        self.input_project = get_default_feed_forward_operator(in_channels + 2, embed_dim, hidden_channels=embed_dim)

    def forward(self, v, domain_size_x, domain_size_y):
        assert v.size(1) == self.in_channels
        assert isinstance(domain_size_y, int)
        assert isinstance(domain_size_x, int)
        assert domain_size_y % self.sub_size_y == 0
        assert domain_size_x % self.sub_size_x == 0
        n_sub_y = domain_size_y // self.sub_size_y
        n_sub_x = domain_size_x // self.sub_size_x

        # concatenate point-wise positions
        # TODO: for some problems, this should depend on the domain size...
        # TODO: Ideally, should use a better pos encoding than just positions...
        height = v.size(-2)
        width = v.size(-1)
        g = 2 * cell_centered_unit_grid(
            (height, width), device=v.device
        ) - 1
        g = einops.repeat(g, "... -> b ...", b=v.size(0))
        v = torch.cat((g, v), dim=1)

        d = decompose2d(v, n_sub_x, n_sub_y)
        d = self.input_project(d)
        
        for encoder in self.encoder:
            d = encoder(d, n_sub_x, n_sub_y)

        d = self.output_project(d)
        u = recompose2d(d, n_sub_x, n_sub_y)
        
        return u