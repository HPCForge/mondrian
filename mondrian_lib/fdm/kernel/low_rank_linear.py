import torch
import torch.nn as nn
from neuralop.layers.mlp import MLP
from mondrian_lib.fdm.integral import integral_2d

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

        self.psi = torch.nn.Parameter(torch.empty((self.rank, self.in_channels, 2), requires_grad=True))
        torch.nn.init.kaiming_normal_(self.psi)

        self.phi = torch.nn.Parameter(torch.empty((self.out_channels, self.rank, 2), requires_grad=True))
        torch.nn.init.kaiming_normal_(self.phi)

    def forward(self, v, x):
        r"""
        Integral kernel operator. For each coordinate, m we evaluate the
        operator to all discretization points of v.
        Args:
            v: discretized function [batch, in_size, H, W]
            x: grid of coordinates [H, W, 2]
        Returns
            u: discretized function [batch, out_size, H, W]
        """

        x_res = v.size(3)
        y_res = v.size(2)

        # Just use [-1, 1], since we can assume arbitrary coord system
        #x_coords, y_coords = torch.meshgrid(
        #        torch.linspace(-1, 1, x_res, device=v.device),
        #        torch.linspace(-1, 1, y_res, device=v.device),
        #        indexing='xy')

        # [H, W, 2]
        #x = torch.stack((y_coords, x_coords), dim=-1)

        # [H, W, rank, in_size]
        psi = torch.einsum('rix,hwx->hwri', self.psi, x)
        
        # [H, W, rank, out_size]
        phi = torch.einsum('orx,hwx->hwro', self.phi, x)

        # [batch, H, W, in_size]
        vp = v.permute(0, 2, 3, 1)

        # 1. Evaluate vector inner product for each x
        # [batch, H, W, rank]
        inner = torch.einsum('bhwi,hwri->bhwr', vp, psi)
        
        # 2. Evaluate L2 inner product via double integral 
        # [batch, rank]
        delta_x = x[0, 1, 0] - x[0, 0, 0]
        l2_inner = integral_2d(inner, dx=delta_x, dim1=1, dim2=2)

        u = torch.einsum('br,hwro->bhwo', l2_inner, phi)
        return u.permute(0, 3, 1, 2)
