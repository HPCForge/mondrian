import torch
from torch import nn
import math
from ..grid.utility import cell_centered_unit_grid


class LogCPB(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads, bias=False),
        )

    def _get_grid(self, n_sub_x, n_sub_y, device):
        r"""
        taken and modified from https://github.com/microsoft/Swin-Transformer/
        """
        relative_coords_h = torch.arange(
            -n_sub_y // 2, n_sub_y // 2, dtype=torch.float32, device=device
        )
        relative_coords_w = torch.arange(
            -n_sub_x // 2, n_sub_x // 2, dtype=torch.float32, device=device
        )
        relative_coords_table = (
            torch.stack(
                torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij")
            )
            .contiguous()
            .permute(1, 2, 0)
        )
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / math.log2(8)
        )
        return relative_coords_table

    def _get_bias(self, n_sub_x, n_sub_y, device):
        grid = self._get_grid(n_sub_x, n_sub_y, device)
        flat = grid.view(-1, 2)
        diff = flat.unsqueeze(1) - flat
        return diff

    def forward(self, n_sub_x, n_sub_y, device):
        r"""
        Args:
          n_sub_x: number of subdomains in x direction
          n_sub_y: number of subdomains in y direction
        Returns:
          bias: [1 x num_heads x seq_len x seq_len] is a
                position bias, seq_len == n_sub_x * n_sub_y
        """
        grid = self._get_bias(n_sub_x, n_sub_y, device)
        pos = self.mlp(grid).permute(2, 0, 1).unsqueeze(0)
        return pos


class FixedCPB2d(nn.Module):
    r"""
    This provides a conditional bias for a grid on a fixed-size domain.
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads, bias=False),
        )

    def _get_bias(self, height, width, device):
        grid = cell_centered_unit_grid((height, width), device=device).permute(1, 2, 0)
        flat = grid.view(-1, 2)
        diff = flat.unsqueeze(1) - flat
        return diff

    def forward(self, height, width, device):
        r"""
        Args:
          n_sub_x: number of subdomains in x direction
          n_sub_y: number of subdomains in y direction
        Returns:
          bias: [1 x num_heads x seq_len x seq_len] is a
                position bias, seq_len == height * width
        """
        grid = self._get_bias(height, width, device)
        pos = self.mlp(grid).permute(2, 0, 1).unsqueeze(0)
        return pos
