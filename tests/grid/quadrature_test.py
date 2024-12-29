import pytest
import torch

from mondrian.grid.quadrature import (
    reimann_quadrature_weights,
    trapezoid_quadrature_weights,
    simpsons_13_quadrature_weights,
    Integral2d,
    InnerProduct2d,
)
from mondrian.test_util import available_devices


def exp_cos(d):
    # example taken from Kat Kinson, Chapter 5
    x = torch.linspace(0, torch.pi, d)
    x2 = torch.exp(x) * torch.cos(x)
    return x2


true_exp_cos = -12.0703463164


def double_cos(d):
    x = torch.linspace(0, torch.pi, d)
    y = torch.linspace(0, 1, d // 2)
    x, y = torch.meshgrid(x, y, indexing="xy")
    return torch.exp(x) * torch.cos(x + y)


true_double_cos = -15.70555657084206035


@pytest.mark.parametrize("device", available_devices())
def test_reimann(device):
    w = reimann_quadrature_weights((torch.pi,), (129,), device)
    x2 = exp_cos(129).to(device)
    assert torch.isclose((x2 * w).sum(), torch.tensor(true_exp_cos), atol=5e-1)


@pytest.mark.parametrize("device", available_devices())
def test_trapezoid(device):
    w = trapezoid_quadrature_weights((torch.pi,), (129,), device).float()
    x2 = exp_cos(129).to(device)
    assert torch.isclose((w * x2).sum(), torch.tensor(true_exp_cos), atol=1e-2)


@pytest.mark.parametrize("device", available_devices())
def test_simpsons(device):
    w = simpsons_13_quadrature_weights((torch.pi,), (129,), device).float()
    x2 = exp_cos(129).to(device)
    assert torch.isclose((w * x2).sum(), torch.tensor(true_exp_cos))

    # on an even-sized grid, it uses a mix of simpsons and trapezoid rule on the edge
    # of the grid, so worth testing separately.
    w = simpsons_13_quadrature_weights((torch.pi,), (128,), device).float()
    x2 = exp_cos(128).to(device)
    assert torch.isclose((w * x2).sum(), torch.tensor(true_exp_cos))

    w = simpsons_13_quadrature_weights((1, torch.pi), (129 // 2, 129), device).float()
    x2 = double_cos(129).to(device)
    assert torch.allclose((w * x2).sum(), torch.tensor(true_double_cos))
