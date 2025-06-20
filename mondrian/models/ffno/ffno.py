r"""
Taken and modified from Factorized FNO
https://github.com/alasdairtran/fourierflow/
"""

from functools import partial

import torch
import torch.nn as nn
from torch.nn.functional import gelu
from einops import rearrange
from neuralop.layers.padding import DomainPadding

from .feedforward import FeedForward

from mondrian.layers.spectral_conv import FactorizedSpectralConv2d

Linear = partial(nn.Conv2d, kernel_size=(1, 1), stride=1, padding=0)

class FNOFactorized2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        hidden_channels,
        domain_padding,
        dropout=0.0,
        in_dropout=0.0,
        n_layers=4,
        factor=2,
        n_ff_layers=2,
        layer_norm=False,
        mode="full",
    ):
        super().__init__()
        self.modes = modes
        self.width = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers

        self.spectral_layers = nn.ModuleList(
            [
                FactorizedSpectralConv2d(
                    in_dim=hidden_channels,
                    out_dim=hidden_channels,
                    n_modes=modes,
                    mode=mode,
                )
                for _ in range(n_layers)
            ]
        )

        self.skip_layers = nn.ModuleList(
            [
                FeedForward(hidden_channels, factor, n_ff_layers, layer_norm, dropout)
                for _ in range(n_layers)
            ]
        )

        self.lifting = nn.Sequential(
            Linear(in_channels, 128), nn.GELU(), Linear(128, hidden_channels)
        )

        self.projection = nn.Sequential(
            Linear(hidden_channels, 128), nn.GELU(), Linear(128, out_channels)
        )

        self.padding = DomainPadding(domain_padding)

    def forward(self, x):
        x = self.lifting(x)
        x = self.in_drop(x)
        x = self.padding.pad(x)
        for i in range(self.n_layers):
            x = gelu(self.spectral_layers[i](x)) + self.skip_layers[i](x)
        x = self.padding.unpad(x)
        forecast = self.projection(x)
        return forecast
