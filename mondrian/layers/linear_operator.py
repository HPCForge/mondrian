import math

import einops
import torch
from torch import nn

from ..grid.utility import cell_centered_unit_grid
from ..grid.quadrature import get_unit_quadrature_weights

class NeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__(
            LinearOperator2d(in_channels, hidden_channels, bias=True),
            nn.GELU(),
            LinearOperator2d(hidden_channels, out_channels, bias=True),
        )

class SeparableNeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__(
            SeparableLinearOperator2d(in_channels, hidden_channels, bias=True),
            nn.GELU(),
            SeparableLinearOperator2d(hidden_channels, out_channels, bias=True),
        )

class LowRankNeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__(
            LowRankLinearOperator2d(in_channels, hidden_channels, bias=True),
            nn.GELU(),
            LowRankLinearOperator2d(hidden_channels, out_channels, bias=True),
        )

class LinearOperator2dBase(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, 1, 1))
            stdv = 1 / math.sqrt(self.in_channels)
            with torch.no_grad():
                self.bias.uniform_(-stdv, stdv)
        else:
            self.bias = None

class LinearOperator2d(LinearOperator2dBase):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__(in_channels, out_channels, bias)
        self.kernel = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, in_channels * out_channels)
        )
        
    def forward(self, v):
        height = v.size(-2)
        width = v.size(-1)
        coords = cell_centered_unit_grid((height, width, height, width), device=v.device).permute(1, 2, 3, 4, 0)
        kernel = self.kernel(coords).reshape(height, width, height, width, self.out_channels, self.in_channels)
        quadrature_weights = get_unit_quadrature_weights((height, width), device=v.device)
        v = einops.einsum(kernel, quadrature_weights * v, 'h1 w1 h2 w2 o i, ... i h1 w2 -> ... o h2 w2')
        if self.bias is not None:
            v = v + self.bias
        return v

class SeparableLinearOperator2d(LinearOperator2dBase):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__(in_channels, out_channels, bias)
        self.kernel = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, in_channels)
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        
    def forward(self, v):
        height = v.size(-2)
        width = v.size(-1)
        coords = cell_centered_unit_grid((height, width, height, width), device=v.device).permute(1, 2, 3, 4, 0)
        kernel = self.kernel(coords).reshape(height, width, height, width, self.in_channels)
        quadrature_weights = get_unit_quadrature_weights((height, width), device=v.device)
        v = einops.einsum(kernel, quadrature_weights * v, 'h1 w1 h2 w2 i, ... i h1 w2 -> ... i h2 w2')
        v = self.conv(v)
        if self.bias is not None:
            v = v + self.bias
        return v

class LowRankLinearOperator2d(LinearOperator2dBase):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__(in_channels, out_channels, bias)
        self.rank = 16
        self.kernel1 = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, out_channels * self.rank)
        )
        
        self.kernel2 = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, self.rank * in_channels)
        )
        
    def forward(self, v):
        height = v.size(-2)
        width = v.size(-1)
        coords = cell_centered_unit_grid((height, width), device=v.device).permute(1, 2, 0)
        left_kernel = self.kernel1(coords).reshape(height, width, self.out_channels, self.rank)
        right_kernel = self.kernel2(coords).reshape(height, width, self.rank, self.in_channels) 
        quadrature_weights = get_unit_quadrature_weights((height, width), device=v.device)
                    
        rk = einops.einsum(right_kernel, quadrature_weights * v, 'h w r i, ... i h w -> ... r')
        v = einops.einsum(left_kernel, rk, 'h w o r, ... r -> ... o h w')
        
        if self.bias is not None:
            v = v + self.bias
        return v