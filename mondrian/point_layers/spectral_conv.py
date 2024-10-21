import torch
from torch import nn
from neuralop.layers.spectral_convolution import SpectralConv

class SimpleSpectralConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_modes,
                 version='ffno'):
        super().__init__()
        assert version in ('fno', 'ffno')
        if version == 'fno':
            Spectral = SpectralConv
            n_modes = (n_modes, n_modes)
        elif version == 'ffno':
            Spectral = FactorizedSpectralConv

        # FFNO's SpectralConv2d can't project to different channel sizes,
        # so just apply a convolution point-wise after it
        self.spectral = Spectral(in_channels, in_channels, n_modes)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.spectral(x)
        x = self.conv(x)
        return x
