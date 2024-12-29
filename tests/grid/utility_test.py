import pytest
import torch

from mondrian.grid.utility import grid, cell_centered_grid
from mondrian.test_util import available_devices


@pytest.mark.parametrize("device", available_devices())
def test_grid(device):
    g = grid((128, 128), [4, 4], device=device)
    assert g.size(0) == 2
    assert g.size(1) == 128
    assert g.size(2) == 128
    assert g[0][0][0] == -2


@pytest.mark.parametrize("device", available_devices())
def test_cell_centered_grid(device):
    g = cell_centered_grid((128, 128), [4, 4], device=device)
    assert g.size(0) == 2
    assert g.size(1) == 128
    assert g.size(2) == 128
