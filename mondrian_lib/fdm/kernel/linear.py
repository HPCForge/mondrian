import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLP

class LinearKernel(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 rank):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.rank = rank

        hidden_size = 32

        self.L = nn.Linear(self.in_size, self.out_size)

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

        # [2, y_res *  x_res]
        coords = torch.stack((x_coords, y_coords), dim=0)
        diff = coords.unsqueeze(1).unsqueeze(1) - coords.unsqueeze(-1).unsqueeze(-1)

        print(diff.size())

        # [y_res, x_res, y_res, x_res]
        dist = torch.sqrt((diff * diff).sum(0))
        
        # [batch, y_res, x_res, in_size]
        vp = v.permute(0, 2, 3, 1)

        # [batch, y_res * x_res, out_size]
        t = self.L(vp).reshape((v.size(0), -1, self.out_size))

        # [y_res, x_res, y_res * x_res]
        dist = dist.reshape((y_res, x_res, -1))

        up = torch.einsum('bdo,yxd->byxo', t, dist)
        u = up.permute(0, 3, 1, 2)
        return u
