from typing import Tuple
import torch


def is_power_of_2(x: int):
    assert isinstance(x, int)
    # gross way of checking number of 1 bits...,
    return bin(x).count("1") == 1


def grid(grid_disc: Tuple[int], grid_size: Tuple[int], device):
    assert len(grid_disc) == len(grid_size)
    center = [(-(g / 2), g / 2) for g in grid_size]
    coords = [
        torch.linspace(center[i][0], center[i][1], grid_disc[i], device=device)
        for i in range(len(grid_size))
    ]
    # ij indexing since tuples goes z, y, x
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=0)


def cell_centered_grid(grid_disc: Tuple[int], grid_size: Tuple[int], device):
    assert len(grid_disc) == len(grid_size)
    dims = []
    for dim in range(len(grid_disc)):
        dim_size = grid_size[dim]
        dim_disc = grid_disc[dim]
        delta_x = dim_size / (dim_disc + 1)
        dim_points = (
            delta_x * (torch.arange(dim_disc, device=device) + 0.5) - dim_size / 2
        )
        dims.append(dim_points)
    # ij indexing since tuples goes z, y, x
    return torch.stack(torch.meshgrid(*dims, indexing="ij"), dim=0)
