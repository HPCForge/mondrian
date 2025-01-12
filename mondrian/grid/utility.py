import functools
import math
from typing import Tuple
import torch

from .decompose import decompose2d, recompose2d

def is_power_of_2(x: int):
    assert isinstance(x, int)
    # gross way of checking number of 1 bits...,
    return bin(x).count("1") == 1

@functools.lru_cache(maxsize=10)
def grid(grid_disc: Tuple[int], grid_size: Tuple[int], zero_mean: bool, device):
    assert len(grid_disc) == len(grid_size)
    dims = []
    for dim in range(len(grid_disc)):
        dim_points = torch.linspace(0, grid_size[dim], grid_disc[dim], device=device)
        if zero_mean:
            dim_points = dim_points - (grid_size[dim] / 2)
        dims.append(dim_points)
    if len(grid_disc) == 1:
        return dims[0].unsqueeze(0)
    return torch.stack(torch.meshgrid(*dims, indexing="ij"), dim=0)

def unit_grid(grid_disc, device):
    grid_size = tuple([1 for _ in range(len(grid_disc))])
    return grid(grid_disc, grid_size, zero_mean=False, device=device)

@functools.lru_cache(maxsize=10)
def cell_centered_grid(grid_disc: Tuple[int], grid_size: Tuple[int], zero_mean, device):
    r"""
    Cell-centered grid, [0, grid_size[0]] x [0, grid_size[1]] x ...
    Args:
        grid_disc: tuple of number of points in each dimension
        grid_size: tuple of grid dimensions' physical sizes.
        zero_mean: boolean, whether to make grid centered around zero
        device: device to create grid on
    """
    assert len(grid_disc) == len(grid_size)
    dims = []
    for dim in range(len(grid_disc)):
        dim_size = grid_size[dim]
        dim_disc = grid_disc[dim]
        delta_x = dim_size / dim_disc
        dim_points = torch.arange(dim_disc, device=device) + 0.5
        if zero_mean:
            dim_points = dim_points - (dim_size / 2)
        dim_points = delta_x * dim_points
        dims.append(dim_points)
    if len(grid_disc) == 1:
        return dims[0]
    # ij indexing since tuples goes z, y, x
    return torch.stack(torch.meshgrid(*dims, indexing="ij"), dim=0)

@functools.lru_cache(maxsize=10)
def cell_centered_unit_grid(grid_disc: Tuple[int], device):
    r"""
    cell-centered grid on cube [0, 1]^len(grid_disc)
    """
    dims = []
    for dim in range(len(grid_disc)):
        dim_disc = grid_disc[dim]
        delta_x = 1 / dim_disc
        dim_points = (
            delta_x * (torch.arange(dim_disc, device=device) + 0.5)
        )
        dims.append(dim_points)
    if len(grid_disc) == 1:
        return dims[0]
    # ij indexing since tuples goes z, y, x
    return torch.stack(torch.meshgrid(*dims, indexing="ij"), dim=0)

@torch.compiler.disable
def interpolate_sequence(seq, height, width):
    r"""
    Interpolating subdomains is sort of odd...
    """
    # This just assumes it's a square atm... 
    n_sub = int(math.sqrt(seq.size(1)))
    data = recompose2d(seq, n_sub, n_sub)
    data = torch.nn.functional.interpolate(data, 
                                           size=(n_sub * height, n_sub * width),
                                           mode='nearest-exact', 
                                           #align_corners=False
                                           )
    seq = decompose2d(data, n_sub, n_sub)
    return seq