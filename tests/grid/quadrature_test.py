import pytest
import torch

from mondrian.grid.quadrature import (
    reimann_quadrature_weights,
    trapezoid_quadrature_weights,
    simpsons_13_quadrature_weights,
)
from mondrian.test_util import available_devices


def exp_cos(d):
    # example taken from Kat Kinson, Chapter 5
    # NOTE: This is a cell-centered grid, which is
    # slightly different from the kat-kinson, which is vertex centered.
    delta = torch.pi / d
    x = delta * (torch.arange(0, d) + 0.5)
    x2 = torch.exp(x) * torch.cos(x)
    return x2

true_exp_cos = -12.0703463164

def double_cos(d):
    delta_x = torch.pi / d
    delta_y = 1 / (d // 2)
    x = delta_x * (torch.arange(0, d) + 0.5)
    y = delta_y * (torch.arange(0, d // 2) + 0.5)
    x, y = torch.meshgrid(x, y, indexing="xy")
    return torch.cos(x * y)

true_double_cos = 1.851

def mathworld1(d):
    delta = 1 / d
    x = delta * (torch.arange(0, d) + 0.5)
    y = delta * (torch.arange(0, d) + 0.5)
    x, y = torch.meshgrid(x, y, indexing="xy")
    return (x-1)/((1-x*y)*torch.log(x*y))

true_mathworld_1 = 0.577215664901532860606512090082


@pytest.mark.parametrize("device", available_devices())
def test_reimann(device):
    w = reimann_quadrature_weights((torch.pi,), (129,), device)
    x2 = exp_cos(129).to(device)
    assert torch.isclose((x2 * w).sum(), torch.tensor(true_exp_cos), atol=5e-1)
    
    w = reimann_quadrature_weights((1, torch.pi), (129 // 2, 129), device).float()
    x2 = double_cos(129).to(device)
    assert torch.allclose((w * x2).sum(), torch.tensor(true_double_cos), atol=1e-3)

@pytest.mark.parametrize("device", available_devices())
def test_trapezoid_1d(device):
    w = trapezoid_quadrature_weights((torch.pi,), (129,), device).float()
    x2 = exp_cos(129).to(device)
    assert torch.isclose((w * x2).sum(), torch.tensor(true_exp_cos), atol=1e-2)

    w = trapezoid_quadrature_weights((torch.pi,), (128,), device).float()
    x2 = exp_cos(128).to(device)
    assert torch.isclose((w * x2).sum(), torch.tensor(true_exp_cos), atol=1e-2)

@pytest.mark.parametrize("device", available_devices())
def test_trapezoid_2d(device):
    w = trapezoid_quadrature_weights((1, torch.pi), (129 // 2, 129), device).float()
    x2 = double_cos(129).to(device)
    assert torch.allclose((w * x2).sum(), torch.tensor(true_double_cos), atol=1e-2)

@pytest.mark.parametrize("device", available_devices())
def test_simpsons_1d(device):
    w = simpsons_13_quadrature_weights((torch.pi,), (129,), device).float()
    x2 = exp_cos(129).to(device)
    assert torch.isclose((w * x2).sum(), torch.tensor(true_exp_cos))

    # on an even-sized grid, it uses a mix of simpsons and trapezoid rule on the edge
    # of the grid, so worth testing separately.
    w = simpsons_13_quadrature_weights((torch.pi,), (128,), device).float()
    x2 = exp_cos(128).to(device)
    assert torch.isclose((w * x2).sum(), torch.tensor(true_exp_cos))

"""
@pytest.mark.parametrize("device", available_devices())
def test_simpsons_2d(device):
    w = simpsons_13_quadrature_weights((1, torch.pi), (129 // 2, 129), device).float()
    x2 = double_cos(129).to(device)
    assert torch.allclose((w * x2).sum(), torch.tensor(true_double_cos), atol=1e-3)
"""