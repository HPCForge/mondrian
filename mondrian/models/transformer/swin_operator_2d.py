from typing import Union, Tuple

import torch
from torch import nn
import einops

from mondrian.grid.decompose import decompose2d, recompose2d
from mondrian.grid.attention.vector_self_attention import VectorSelfAttention
from mondrian.layers.pointwise import PointwiseMLP2d
from mondrian.layers.learned_pos_embedding import LearnedPosEmbedding2d
from mondrian.grid.utility import grid

from .mlp import MLP


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = VectorSelfAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, embed_dim, embed_dim, num_layers=2)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class SwinOperator2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        head_split: str,
        score_method: str,
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
            [Encoder(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.input_project = PointwiseMLP2d(
            in_channels + 2, embed_dim, hidden_channels=128
        )
        self.output_project = PointwiseMLP2d(embed_dim, out_channels, hidden_channels=128)

        # TODO: Maybe make this optional...
        self.pos_embedding = LearnedPosEmbedding2d(
            max_seq_len=max_seq_len, channels=embed_dim
        )

    def _interpolate(self, x, target_height, target_width):
        return nn.functional.interpolate(
            x, size=(target_height, target_width), mode="bilinear"
        )

    def _get_coarsities_and_n_sub(self, height, width, n_sub_y, n_sub_x):
        num_down_layers = self.num_layers // 2
        num_up_layers = self.num_layers - num_down_layers

        down_layer_coarsity = [
            (height // (2**i), width // (2**i)) for i in range(num_down_layers)
        ]
        up_layer_coarsity = [
            (height // (2**i), width // (2**i)) for i in range(num_down_layers)[::-1]
        ]
        if num_down_layers != num_up_layers:
            up_layer_coarsity.append((height, width))

        coarsity = down_layer_coarsity + up_layer_coarsity

        down_layer_n_sub = [
            (n_sub_y // (2**i), n_sub_x // (2**i)) for i in range(num_down_layers)
        ]
        up_layer_n_sub = [
            (n_sub_y // (2**i), n_sub_x // (2**i)) for i in range(num_down_layers)[::-1]
        ]
        if num_down_layers != num_up_layers:
            up_layer_n_sub.append((n_sub_y, n_sub_x))

        n_sub = down_layer_n_sub + up_layer_n_sub

        return coarsity, n_sub

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

        height = x.size(-2)
        width = x.size(-1)

        # concatenate point-wise positions
        g = grid((height, width), (domain_size_y, domain_size_x)).to(x.device)
        g = einops.repeat(g, "... -> b ...", b=x.size(0))
        x = torch.cat((g, x), dim=1)

        v = self.input_project(v)
        d = decompose2d(v, n_sub_x, n_sub_y)
        d = self.pos_embedding(d)
        d = recompose2d(v, n_sub_x, n_sub_y)

        coarsity, n_subs = self._get_coarsities_and_n_sub(height, width, n_sub_y, n_sub_x)

        for (height, width), (n_sub_x, n_sub_y), encoder in zip(
            coarsity, n_subs, self.encoder
        ):
            # decompose and apply attention inside subdomains
            d = decompose2d(d, n_sub_x, n_sub_y)
            d = encoder(d, n_sub_x, n_sub_y)
            # recompose the domains and coarsen the total grid.
            d = recompose2d(d, n_sub_x, n_sub_y)
            d = self._interpolate(d, height, width)

        # u = recompose2d(d, n_sub_x, n_sub_y)
        u = self.output_project(u)

        return u
