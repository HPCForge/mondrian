import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLP

class NystromLinearKernel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 sample_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.sample_size = sample_size

        self.weight = torch.empty((self.out_channels, self.in_channels, 2))
        torch.nn.init.kaiming_normal(self.weight)

    def forward(self, v):
        r"""
        Integral kernel operator. For each coordinate, m we evaluate the
        operator to all discretization points of v.
        Args:
            v: discretized function [batch, in_size, H, W]
        Returns
            u: discretized function [batch, out_size, H, W]
        """
        assert v.dim() == 4

        batch_size = v.size(0)
        x_res = v.size(3)
        y_res = v.size(2)

        # Just use [0-1], since we can assume arbitrary coord system
        x_coords, y_coords = torch.meshgrid(
                torch.linspace(-1, 1, x_res, device=v.device),
                torch.linspace(-1, 1, y_res, device=v.device),
                indexing='xy')

        # [y_res, x_res, 2]
        coords = torch.stack((x_coords, y_coords), dim=-1).reshape((-1, 2))

        # [batch, J, in_channels]
        vp = v.permute(0, 2, 3, 1).reshape(v.size(0), J, self.in_channels)

        # select a random sample
        indices = torch.randperm(J)[:self.sample_size]
        # [J', 2]
        sample_coords = coords[indices]
        # [batch, J', in_size]
        sample_vp = vp[:, indices]
        
        # [J, J', 2]
        x = coords.unsqueeze(1).repeat(1, self.sample_size, 1)
        y = sample_coords.unsqueeze(0).repeat(J, 1, 1)

        # [J, J', 2]
        diff = (x - y)

        # for each target and sample point, we build a matrix
        # by contracting along the coordinate axis
        weight = torch.einsum('oic,rsc->rsoi', self.weight, diff)

        # [batch, out_channels, J, J']
        u = torch.einsum('rsoi,bsi->bors', weight, sample_vp).sum(-1)
        return u.reshape(batch_size, self.out_channels, y_res, x_res)
