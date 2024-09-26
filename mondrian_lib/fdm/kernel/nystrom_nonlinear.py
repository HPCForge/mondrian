import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLP
import einops

class NystromNonLinearKernel(nn.Module):
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
        self.n_dim = 2

        self.grid = torch.empty((self.sample_size, self.sample_size, 2))
        self.grid.uniform_(-1, 1)
        # TODO: check what this is doing :-)
        self.grid, _ = self.grid.sort(dim=1)
        self.grid, _ = self.grid.sort(dim=2)

        #self.kernel = MLP(
        #    in_channels=self.n_dim,
        #    out_channels=self.in_channels * self.out_channels,
        #    hidden_channels=self.hidden_channels * 2,
        #    n_layers=self.n_dim,
        #    n_dim=self.n_dim,
        #)
        self.kernel = nn.Sequential(
            nn.Linear(self.n_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.in_channels * self.out_channels)
        )

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
        self.grid = self.grid.to(v.device)

        batch_size = v.size(0)
        x_res = v.size(3)
        y_res = v.size(2)
        J = x_res * y_res

        row_indices = torch.randperm(J)[:self.sample_size]
        col_indices = torch.randperm(J)[:self.sample_size]

        #x_coords, y_coords = torch.meshgrid(
        #        torch.linspace(-1, 1, x_res, device=v.device),
        #        torch.linspace(-1, 1, y_res, device=v.device),
        #        indexing='xy')

        # [H, W, 2]
        #coords = torch.stack((x_coords, y_coords), dim=-1)

        # [H, W, 1, 1, 2]
        grid_tgt = self.grid.unsqueeze(2).unsqueeze(2)
        
        # [H, W, H_sample, W_sample, 2]
        diff = self.grid - grid_tgt 

        # [B, H_sample, H_sample, 2]
        sample_grid = self.grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # [B, in_size, H_sample, W_sample]
        sample = F.grid_sample(v, sample_grid)

        m = self.kernel(diff).unflatten(dim=-1, sizes=(self.out_channels, self.in_channels))
        out = einops.einsum(m, sample, 'h1 h2 h3 h4 m n, b n h3 h4 -> b n h1 h2') / self.sample_size

        return F.interpolate(out, size=(y_res, x_res), mode='bilinear') 
