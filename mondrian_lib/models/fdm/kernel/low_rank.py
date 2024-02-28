import torch
import torch.nn as nn
from neuralop.layers.mlp import MLP

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

        hidden_size = 64

        self.psi = nn.Sequential(
                nn.Linear(2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.rank * self.in_size))

        self.phi = nn.Sequential(
                nn.Linear(2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.rank * self.out_size))

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

        # Just use [0-1], since we can assume arbitrary coord system
        x_coords, y_coords = torch.meshgrid(
                torch.linspace(0, 1, x_res, device=v.device),
                torch.linspace(0, 1, y_res, device=v.device),
                indexing='xy')

        # Currently assume uniform discretization
        delta_x = x_coords[0, 1] - x_coords[0, 0]

        # [H, W, 2]
        x = torch.stack((x_coords, y_coords), dim=-1)

        # [H, W, rank, in_size]
        psi = self.psi(x).reshape((y_res, x_res, self.rank, self.in_size))
        
        # [H, W, rank, out_size]
        phi = self.phi(x).reshape((y_res, x_res, self.rank, self.out_size))

        # [batch, H, W, 1, n]
        vp = v.permute(0, 2, 3, 1).unsqueeze(3)
        # [batch, H, W, r, 1]
        dot = (vp * psi * delta_x).sum(-1).unsqueeze(-1)

        # [batch, H, W, m]  
        u = (dot * phi).sum(3)
        # [batch, m, H, W]  
        return u.permute(0, 3, 1, 2)

        """
        for r in range(self.rank):
            # [H, W, in_size]
            psi_r = self.psi[r](x)
            # [H, W, out_size]
            phi_r = self.phi[r](x)

            # [batch, H, W, in_size]
            vp = v.permute(0, 2, 3, 1)
            # [batch, H, W, 1]
            dot = (psi_r * vp * delta_x).sum(-1).unsqueeze(-1)
            u_r = dot * phi_r
            u += u_r
        """
        return u.permute(0, 3, 1, 2)
