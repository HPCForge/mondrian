import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from neuralop.layers.spectral_convolution import SpectralConv

from mondrian.layers.seq_op import seq_op

VERSIONS = ["fno", "ffno", "cnn", "random"]

_DEFAULT_SPECTRAL_CONV_MODES = 0

def set_default_spectral_conv_modes(modes: int):
    global _DEFAULT_SPECTRAL_CONV_MODES
    logging.info(f'changing _DEFAULT_SPECTRAL_CONV_MODES to {modes}')
    _DEFAULT_SPECTRAL_CONV_MODES = modes

class SpectralConvNeuralOperator(nn.Sequential):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 hidden_channels,
                 n_modes,
                 version="fno"):
        super().__init__(
            SimpleSpectralConv2d(in_channels, hidden_channels, n_modes=n_modes, version=version),
            nn.GELU(),
            SimpleSpectralConv2d(hidden_channels, out_channels, n_modes=n_modes, version=version)
        )

class SimpleSpectralConv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n_modes,
                 bias: bool = True, 
                 version: str = "fno"):
        super().__init__()
        assert version in VERSIONS
        assert n_modes > 0
        self.in_channels = in_channels
        if version == "fno":
            n_modes = (n_modes, n_modes)
            self.spectral = SpectralConv(in_channels, out_channels, n_modes)
        elif version == "ffno":
            n_modes = n_modes // 2
            self.spectral = FactorizedSpectralConv2d(in_channels, out_channels, n_modes)
        elif version == "cnn":
            # This is just for testing
            self.spectral = nn.Conv2d(
                in_channels, out_channels, kernel_size=(7, 7), padding=3
            )
        elif version == "random":
            self.spectral = RandomProjectionConv2d(
                in_channels, out_channels, inner_dim=32
            )
            
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, 1, 1))
            stdv = 1 / math.sqrt(self.in_channels)
            with torch.no_grad():
                self.bias.uniform_(-stdv, stdv)
        else:
            self.bias = None

    # compile doesn't like complex-valued stuff.
    @torch.compiler.disable
    def forward(self, x):
        r"""
        Args:
            x: [batch x seq x in_channels x H x W]
        Returns:
            out: [batch x seq x out_channels x H x W]
        """
        assert x.dim() == 5
        x = seq_op(self.spectral, x)
        if self.bias is not None:
            x = x + self.bias
        return x

class FactorizedSpectralConv2d(nn.Module):
    r"""
    Taken from [fourierflow](https://github.com/alasdairtran/fourierflow).
    1. This was pretty spaghetti'd so this removes the non-critical portions.
    2. Use out_dim in fourier_weight, so it can project to different channel sizes.

    Autograd graph seems to get huge in this...
    """

    def __init__(self, in_dim, out_dim, n_modes, mode="full"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode

        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        self.fourier_weight = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(in_dim, out_dim, n_modes, 2))
                for _ in range(2)
            ]
        )

        for i in range(len(self.fourier_weight)):
            nn.init.xavier_normal_(self.fourier_weight[i])

    def forward(self, x):
        B, I, M, N = x.shape

        # apply spectral conv along y-dimension
        # [batch_size, in_dim, grid_size, grid_size // 2 + 1]
        x_fty = torch.fft.rfft(x, dim=-1, norm="ortho")
        # [batch_size, out_dim, grid_size, grid_size // 2 + 1, 2]
        out_ft = x_fty.new_zeros(B, self.out_dim, M, N // 2 + 1)
        out_ft[:, :, :, : self.n_modes] = einsum(
            x_fty[:, :, :, : self.n_modes],
            torch.view_as_complex(self.fourier_weight[0]),
            "b i x y, i o y -> b o x y",
        )
        # [batch_size, in_dim, grid_size, grid_size]
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm="ortho")

        # apply spectral conv along x-dimension
        # [batch_size, in_dim, grid_size // 2 + 1, grid_size]
        x_ftx = torch.fft.rfft(x, dim=-2, norm="ortho")        
        # [batch_size, out_dim, grid_size // 2 + 1, grid_size, 2]
        out_ft = x_ftx.new_zeros(B, self.out_dim, M // 2 + 1, N)
        out_ft[:, :, : self.n_modes, :] = einsum(
            x_ftx[:, :, : self.n_modes, :],
            torch.view_as_complex(self.fourier_weight[1]),
            "b i x y, i o x -> b o x y",
        )
        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm="ortho")
        
        return xx + xy


class RandomProjectionConv2d(nn.Module):
    r"""
    This is a 'stupid' idea for something like a spectral convolution.
    It is based on FNO essentially projecting to a finite dimensional space, and relying
    on the skip connections to work like a neural operator.
    This seems to avoid some of the issues with computing an FFT on an overly coarse grid.
    This basically just materializes a finite set of random, simple functions to project to a
    finite vector space, applies the weights, and projects back to a (finite) function space.
    Again, working as a neural operator requires skip connection since information is
    lost when going to a finite space.

    The implementation does something like FFNO, where it applies weights along the
    x- and y-dimensions separately.
    """

    def __init__(self, in_dim, out_dim, inner_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inner_dim = inner_dim
        self.order = 8

        kernel_size = 1
        padding = (kernel_size - 1) // 2
        self.c_x = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)

        self.down_x_func = nn.Parameter(
            torch.randn(self.order, self.inner_dim), requires_grad=False
        )
        self.down_y_func = nn.Parameter(
            torch.randn(self.order, self.inner_dim), requires_grad=False
        )
        self.up_x_func = nn.Parameter(
            torch.randn(self.order, self.inner_dim), requires_grad=False
        )
        self.up_y_func = nn.Parameter(
            torch.randn(self.order, self.inner_dim), requires_grad=False
        )

    def _get_down_x(self, res):
        return F.interpolate(
            self.down_x_func.unsqueeze(0).unsqueeze(0),
            size=(res, self.inner_dim),
            mode="bilinear",
        ).squeeze()

    def _get_down_y(self, res):
        return F.interpolate(
            self.down_y_func.unsqueeze(0).unsqueeze(0),
            size=(res, self.inner_dim),
            mode="bilinear",
        ).squeeze()

    def _get_up_x(self, res):
        return F.interpolate(
            self.up_x_func.unsqueeze(0).unsqueeze(0),
            size=(res, self.inner_dim),
            mode="bilinear",
        ).squeeze()

    def _get_up_y(self, res):
        return F.interpolate(
            self.up_y_func.unsqueeze(0).unsqueeze(0),
            size=(res, self.inner_dim),
            mode="bilinear",
        ).squeeze()

    def forward(self, x):
        assert x.dim() == 4

        res_x, res_y = x.size(-1), x.size(-2)

        dpx = self._get_down_x(res_x)
        dpy = self._get_down_y(res_y)

        down_x = einsum(x, dpx, "b c y x, x i -> b c y i")  / res_x
        down_yx = einsum(down_x, dpy, "b c y i, y j -> b c j i") / res_y

        scale_yx = self.c_x(down_yx)

        upx = self._get_up_x(res_x)
        upy = self._get_up_y(res_y)

        new_x = einsum(scale_yx, upx, "b oc j i, x i -> b oc j x")# / res_x
        new_yx = einsum(new_x, upy, "b oc j x, y j -> b oc y x")# / res_y

        return new_yx