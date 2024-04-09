import torch
import torch.nn as nn
from neuralop.layers.mlp import MLP

class LowRankLinearKernel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 rank):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank

        self.psi = torch.empty((self.rank, self.in_channels, 2), requires_grad=True)
        torch.nn.init.kaiming_normal(self.psi)

        self.phi = torch.empty((self.out_channels, self.rank, 2), requires_grad=True)
        torch.nn.init.kaiming_normal(self.phi)

    def forward(self, v):
        r"""
        Integral kernel operator. For each coordinate, m we evaluate the
        operator to all discretization points of v.
        Args:
            v: discretized function [batch, in_size, H, W]
        Returns
            u: discretized function [batch, out_size, H, W]
        """

        x_res = v.size(3)
        y_res = v.size(2)

        # Just use [-1, 1], since we can assume arbitrary coord system
        x_coords, y_coords = torch.meshgrid(
                torch.linspace(-1, 1, x_res, device=v.device),
                torch.linspace(-1, 1, y_res, device=v.device),
                indexing='xy')

        # [H, W, 2]
        x = torch.stack((y_coords, x_coords), dim=-1)

        # [H, W, rank, in_size]
        psi = torch.einsum('rix,hwx->hwri', self.psi, x)
        
        # [H, W, rank, out_size]
        phi = torch.einsum('orx,hwx->hwro', self.phi, x)

        # [batch, H, W, 1, in_size]
        vp = v.permute(0, 2, 3, 1).unsqueeze(3)

        # Currently assume uniform discretization
        delta_x = x_coords[0, 1] - x_coords[0, 0]

        # Approximate L2 inner product <vp, psi>_L2(D, R^n)
        # [batch, rank]
        inner = (vp * psi * delta_x).sum(-1).sum([1, 2])

        u = torch.einsum('br,hwro->bhwo', inner, phi)

        return u.permute(0, 3, 1, 2)
