import torch
import torch.nn as nn
from neuralop.layers.mlp import MLP

from mondrian_lib.fdm.integral import integral_2d

class LowRankKernel(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 rank):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.rank = rank

        self.psi = nn.Sequential(
                nn.Linear(2, 256),
                nn.GELU(),
                #nn.Linear(256, 256),
                #nn.GELU(),
                nn.Linear(256, self.rank * self.in_size))

        self.phi = nn.Sequential(
                nn.Linear(2, 256),
                nn.GELU(),
                #nn.Linear(256, 256),
                #nn.GELU(),
                nn.Linear(256, self.rank * self.out_size))

    def forward(self, v, coords):
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
        #x_coords, y_coords = torch.meshgrid(
        #        torch.linspace(-1, 1, x_res, device=v.device),
        #        torch.linspace(-1, 1, y_res, device=v.device),
        #        indexing='xy')

        # [H, W, 2]
        #x = torch.stack((x_coords, y_coords), dim=-1)

        # [H, W, rank, in_size]
        psi = self.psi(coords).reshape((y_res, x_res, self.rank, self.in_size))
        
        # [H, W, rank, out_size]
        phi = self.phi(coords).reshape((y_res, x_res, self.rank, self.out_size))

        # [batch, H, W, in_size]
        vp = v.permute(0, 2, 3, 1)

        # compute inner products for each x
        inner = torch.einsum('hwri,bhwi->bhwr', psi, vp)

        # integrate over inner products
        # [batch, r]
        dx = coords[0, 1, 0] - coords[0, 0, 0]
        l2_inner = integral_2d(inner, dx=dx, dim1=1, dim2=2)

        u = torch.einsum('br,hwro->bhwo', l2_inner, phi)

        # [batch, m, H, W]  
        return u.permute(0, 3, 1, 2)
