import math

import einops
import torch
from torch import nn

from mondrian.grid.pointwise import PointwiseMLP2d
from mondrian.grid.utility import is_power_of_2


class BubbleMLEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, backbone_model):
        super().__init__()
        assert is_power_of_2(out_channels)
        self.nuc_embed_dim = int(math.sqrt(out_channels))

        self.mlp = PointwiseMLP2d(
            in_channels=in_channels, out_channels=out_channels, hidden_channels=128
        )

        self.l1 = nn.Linear(1, self.nuc_embed_dim, bias=False)
        self.l2 = nn.Linear(1, self.nuc_embed_dim, bias=False)

        self.backbone_model = backbone_model

    def forward(self, x, nuc, domain_size_x=None, domain_size_y=None):
        r"""
        Converts the sequence of nucleation sites' x positions `nuc` to a vector embedding
        and combine it with the input data `x`.
        Args:
          x: [batch, in_channels, H, W]
          nuc: [batch, S*, 1]
        Returns:
          [batch, out_channels, H, W]
        """
        assert x.size(0) == nuc.size(0)

        batch_size = x.size(0)
        x = self.mlp(x)

        # [batch, S*, nuc_embed_dim]
        n1 = self.l1(nuc)
        n2 = self.l2(nuc)

        # nuc is potentially a nested tensor, which seems to not work with einops.
        # [batch, nuc_embed_dim, nuc_embed_dim]
        encoding = torch.matmul(n1.transpose(-2, -1), n2)
        # flatten to a vector and make broadcastable with x
        encoding = encoding.reshape(
            batch_size, self.nuc_embed_dim * self.nuc_embed_dim, 1, 1
        )

        if domain_size_x is not None and domain_size_y is not None:
            return self.backbone_model(x + encoding, domain_size_x, domain_size_y)
        else:
            return self.backbone_model(x + encoding)
