import torch
from torch import nn

import einops

class PointwiseLinear(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.linear = nn.Linear(in_channels, out_channels)
    
  def forward(self, v):
    r"""
    Args:
      v: [batch_size x channels x ...]
    """
    return einops.einsum(
      self.linear.weight, v, 'o i, b i ... -> b o ...')