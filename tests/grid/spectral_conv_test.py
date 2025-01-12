import torch
from torch.nn.functional import interpolate

from mondrian.layers.spectral_conv import SimpleSpectralConv2d


def get_func(d):
    x = torch.linspace(0, 1, d)
    y = torch.linspace(0, 1, d)
    x, y = torch.meshgrid(x, y, indexing="xy")
    return torch.sin(2 * x + y * x).unsqueeze(0).unsqueeze(0).unsqueeze(0)


def test_spectral():
    operators = ["fno", "ffno"]

    # operators should be fairly robust to changing discretization
    for o in operators:
        spectral = SimpleSpectralConv2d(
            in_channels=1, out_channels=1, n_modes=32, version=o
        )
        coarse = spectral(get_func(100)).squeeze(1)
        fine = spectral(get_func(400)).squeeze(1)
        coarse_interp = interpolate(
            coarse, size=(fine.size(-2), fine.size(-1)), mode="bicubic"
        )
        assert ((coarse_interp - fine) ** 2).mean() < 1e-3

    # CNN should have a higher error between coarse and fine, due to discretization changing
    spectral = SimpleSpectralConv2d(
        in_channels=1, out_channels=1, n_modes=32, version="cnn"
    )
    coarse = spectral(get_func(100)).squeeze(1)
    fine = spectral(get_func(400)).squeeze(1)
    coarse_interp = interpolate(
        coarse, size=(fine.size(-2), fine.size(-1)), mode="bicubic"
    )
    assert ((coarse_interp - fine) ** 2).mean() > 1e-3
