from typing import Union, Tuple

import einops
import torch
from torch import nn

from mondrian.attention.func_self_attention import FuncSelfAttention
from mondrian.grid.decompose import decompose2d, recompose2d
from mondrian.layers.seq_op import seq_op
from mondrian.layers.learned_pos_embedding import LearnedPosEmbedding2d
from mondrian.layers.feed_forward_operator import get_default_feed_forward_operator

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
    def __init__(self, embed_dim, channel_heads, x_heads, y_heads, scale, use_bias):
        super().__init__()
        self.sa = FuncSelfAttention(
            embed_dim, channel_heads, x_heads, y_heads, use_bias
        )
        
        self.project = ProjectToNewGrid2d(embed_dim, num_heads=4, scale=scale)
        self.norm1 = SequenceGroupNorm2d(8, embed_dim)
        self.norm2 = SequenceGroupNorm2d(8, embed_dim)

    def forward(self, v, n_sub_x, n_sub_y):
        v = self.sa(self.norm1(v), n_sub_x, n_sub_y) + v
        v = self.project(self.norm2(v))
        return v

class UnetOperator2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        channel_heads: int,
        x_heads: int,
        y_heads: int,
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
        
        # just makes unet structure easier
        assert num_layers % 2 == 0

        self.subdomain_size = subdomain_size
        self.sub_size_y = self.subdomain_size[0]
        self.sub_size_x = self.subdomain_size[1]

        scales = [1] + [0.5 for _ in range(num_layers // 2 - 1)] + [2 for _ in range(num_layers // 2 - 1)] + [1]
        self.encoder = nn.ModuleList(
            [
                Encoder(embed_dim, channel_heads, x_heads, y_heads, scale=scales[layer_idx], use_bias=False)
                for layer_idx in range(num_layers)
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
        d = recompose2d(d, n_sub_x, n_sub_y)
        
        stack = []
        
        scales = [(n_sub_x, n_sub_y)] + [(n_sub_x // i, n_sub_y // i) for i in range(1, len(self.encoder) // 2)]
        
        # TODO: this is completely messed up
        for layer_idx, (encoder) in enumerate(self.encoder):
            if layer_idx < len(self.encoder) // 2: 
                n_sub_x, n_sub_y = scales[layer_idx]
                d = decompose2d(d, n_sub_x, n_sub_y)
                d = encoder(d, n_sub_x, n_sub_y)
                stack.append((d, n_sub_x, n_sub_y))   
            else:
                skip, n_sub_x, n_sub_y = stack.pop()
                d = decompose2d(d, n_sub_x, n_sub_y) 
                d = encoder(d, n_sub_x, n_sub_y) + skip
            print(layer_idx, n_sub_x, n_sub_y)
            d = recompose2d(d, n_sub_x, n_sub_y)
            
        #d = self.output_project(d)
        #u = recompose2d(d, n_sub_x, n_sub_y)
        
        return d