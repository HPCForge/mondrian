import torch
from torch import nn

import einops
from .seq_op import seq_op


class PointwiseLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, v):
        r"""
        Args:
          v: [batch_size x channels x ...]
        """
        return einops.einsum(self.linear.weight, v, "o i, b i ... -> b o ...")

class PointwiseLinear2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        
    def forward(self, v):
        return seq_op(self.linear, v)
        #return einops.einsum(self.linear.weight, v, "o i, ... i h w -> ... o h w")


class PointwiseMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1)),
        )

    def forward(self, v):
        r"""
        Args:
          v: [batch_size x channels x H x W]
        """
        return self.seq(v)
