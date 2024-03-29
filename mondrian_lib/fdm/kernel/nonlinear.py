import torch
import torch.nn as nn
from neuralop.layers.mlp import MLP

class NonLinearKernel(nn.Module):
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
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.rank * self.out_size))

        self.phi = nn.Sequential(
                nn.Linear(2 + self.in_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
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

        # [H, W, rank, out_size]
        psi = self.psi(x).reshape((y_res, x_res, self.rank, self.out_size))

        # [batch, H, W, rank, out_size]
        psi = psi.repeat(v.size(0), 1, 1, 1, 1)

        # [batch, in_size, H, W]
        vp = v.permute(0, 2, 3, 1)

        # [batch, H, W, 2 + in_size]
        xvp = torch.cat((x.repeat(v.size(0), 1, 1, 1), vp), dim=-1)

        # [batch, H, W, rank, out_size]
        phi = self.phi(xvp).reshape((v.size(0), y_res, x_res, self.rank, self.out_size))

        print(psi.size(), phi.size())

        # [batch, H, W, out_size]
        u = (psi * phi).sum(3)

        return u.permute(0, 3, 1, 2)
