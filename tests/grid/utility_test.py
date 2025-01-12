import pytest
import torch

from mondrian.grid.utility import cell_centered_grid
from mondrian.test_util import available_devices



@pytest.mark.parametrize("device", available_devices())
def test_cell_centered_grid(device):
    g = cell_centered_grid((128, 128), (4, 4), zero_mean=False, device=device)
    assert g.size(0) == 2
    assert g.size(1) == 128
    assert g.size(2) == 128


@pytest.mark.parametrize("device", available_devices())
def test_cell_centered_grid(device):
    g = cell_centered_grid((128, 128), (4, 4), zero_mean=True, device=device)
    assert g.size(0) == 2
    assert g.size(1) == 128
    assert g.size(2) == 128
    
    g = cell_centered_grid((128,), (4,), zero_mean=False, device=device)
    # should be equidistant from zero
    assert abs(g[0]) == abs(4 - g[-1])