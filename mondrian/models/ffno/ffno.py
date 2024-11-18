r"""
Taken and modified from Factorized FNO
https://github.com/alasdairtran/fourierflow/
"""

import torch
import torch.nn as nn
from einops import rearrange

from .feedforward import FeedForward
from .linear import WNLinear

from mondrian.grid.spectral_conv import (
    FactorizedSpectralConv2d,
    FeedForward
)

class FNOFactorized2DBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 modes,
                 width,
                 dropout=0.0,
                 in_dropout=0.0,
                 n_layers=4,
                 share_weight: bool = False,
                 share_fork=False,
                 factor=2,
                 ff_weight_norm=False,
                 n_ff_layers=2,
                 gain=1,
                 layer_norm=False,
                 use_fork=False,
                 mode='full'):
        super().__init__()
        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_proj = WNLinear(self.in_channels, width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork

        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = FeedForward(
                    width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = FeedForward(
                width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(width, width, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param, gain=gain)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(FactorizedSpectralConv2d(
                in_dim=width,
                out_dim=width,
                n_modes=modes,
                permute=False,
                forecast_ff=self.forecast_ff,
                backcast_ff=self.backcast_ff,
                fourier_weight=self.fourier_weight,
                factor=factor,
                ff_weight_norm=ff_weight_norm,
                n_ff_layers=n_ff_layers,
                layer_norm=layer_norm,
                use_fork=use_fork,
                dropout=dropout,
                mode=mode))

        self.out = nn.Sequential(
            WNLinear(width, 128, wnorm=ff_weight_norm),
            WNLinear(128, out_channels, wnorm=ff_weight_norm))

    def forward(self, x):
        r"""
        Arthur: For some reason, they expect channels to be last. (maybe for the MLP?)
        I set things up so channels are second dim, which matches
        pytorch's convolutions and neuraloperator. 
        need to permute input and output to get correct dimensions
        """
        x = x.permute((0, 2, 3, 1))

        # x.shape == [n_batches, *dim_sizes, input_size]
        forecast = 0
        x = self.in_proj(x)
        x = self.drop(x)
        forecast_list = []
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b = layer(x)

            #if self.use_fork:
            #    f_out = self.out(f)
            #    forecast = forecast + f_out
            #    forecast_list.append(f_out)

            x = x + b

        if not self.use_fork:
            forecast = self.out(b)

        forecast = forecast.permute((0, 3, 1, 2))

        return forecast
