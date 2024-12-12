from typing import Tuple
import torch


def is_power_of_2(x: int):
    assert isinstance(x, int)
    # gross way of checking number of 1 bits...,
    return bin(x).count("1") == 1


def grid(grid_disc: Tuple[int], grid_size: Tuple[int]):
    assert len(grid_disc) == len(grid_size)
    center = [(-(g / 2), g / 2) for g in grid_size]
    coords = [
        torch.linspace(center[i][0], center[i][1], grid_disc[i])
        for i in range(len(grid_size))
    ]
    return torch.stack(torch.meshgrid(*coords), dim=0)
