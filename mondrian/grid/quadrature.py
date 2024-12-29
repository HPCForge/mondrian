from typing import Tuple

import torch
from torch import nn
import einops

from mondrian.constants import REIMANN, TRAPEZOID, SIMPSON_13


def reimann_quadrature_weights(
    grid_spatial_dimensions: Tuple[int],
    grid_discretization_dimensions: Tuple[int],
    device: torch.device,
):
    r"""
    The arguments should be laid out in z, y, x ordering. For a 2D input,
    you pass the spatial dimensions (height, width) and (yres, xres)
    Args:
      grid_spatial_dimensions: the phyiscal spatial dimensions of a domain.
                               I.e., [1, 1] would be a square domain.
      grid_discretization_dimensions: the number of points used in a discretization along each axis.
                                      I.e., [128, 64] is 128 points in the y direction and 64 in the x direction.
    Returns:
      The quadrature weights that can be used with a Reimann sum. (This just divides each entry by the discretization).
    """
    coords = torch.linspace(
        0, grid_spatial_dimensions[0], grid_discretization_dimensions[0], device=device
    )
    delta = coords[1] - coords[0]
    deltas = torch.tensor(
        [
            delta
            for (grid_axis_spatial, grid_axis_disc) in zip(
                grid_spatial_dimensions, grid_discretization_dimensions
            )
        ]
    )
    delta = torch.prod(deltas)
    weight_sizes = [grid_axis_dim for grid_axis_dim in grid_discretization_dimensions]
    return torch.full(weight_sizes, delta, device=device)


def trapezoid_quadrature_weights(
    grid_spatial_dimensions: Tuple[int],
    grid_discretization_dimensions: Tuple[int],
    device: torch.device,
):
    assert len(grid_spatial_dimensions) == len(grid_discretization_dimensions)
    dims = len(grid_spatial_dimensions)

    weight_sizes = [grid_axis_dim for grid_axis_dim in grid_discretization_dimensions]
    weights = torch.ones(weight_sizes, device=device, dtype=torch.float64)

    for dim in range(dims):

        w = torch.ones(
            grid_discretization_dimensions[dim], device=device, dtype=torch.float64
        )
        w[[0, -1]] = 0.5

        # this makes a tensor of size [1, 1, ..., size, ..., 1, 1]
        # this is so it will be broadcasted along the desired axis of weights.
        size = [1 for _ in range(dims)]
        size[dim] = grid_discretization_dimensions[dim]

        coords = torch.linspace(
            0, grid_spatial_dimensions[dim], grid_discretization_dimensions[dim]
        )
        delta = coords[1] - coords[0]

        # TODO: double check this is right...
        # delta = grid_spatial_dimensions[dim] / grid_discretization_dimensions[dim]
        weights *= delta * w.reshape(size)

    return weights


def simpsons_13_quadrature_weights(
    grid_spatial_dimensions: Tuple[int],
    grid_discretization_dimensions: Tuple[int],
    device,
):
    assert len(grid_spatial_dimensions) == len(grid_discretization_dimensions)
    dims = len(grid_spatial_dimensions)

    weight_sizes = [grid_axis_dim for grid_axis_dim in grid_discretization_dimensions]
    weights = torch.ones(weight_sizes, device=device, dtype=torch.float64)

    for dim in range(dims):

        w = torch.empty(
            grid_discretization_dimensions[dim], device=device, dtype=torch.float64
        )

        if grid_discretization_dimensions[dim] % 2 == 1:
            # if there is an odd number of grid-cells, we can use simpsons directly.
            # [1, 4, 2, 4, 2, ..., 2, 4, 1]
            w[[0, -1]] = 1 / 3
            w[1:-1:2] = 4 / 3
            w[2:-1:2] = 2 / 3
        else:
            # If there is an even number, we need to initialize first odd-part with simpsons,
            # and then have the last partition use the trapezoid rule.
            # [1, 4, 2, 4, 2, ...., 2, 4, 1.5, 0.5]
            w[[0, -2]] = 1 / 3
            w[1:-2:2] = 4 / 3
            w[2:-2:2] = 2 / 3
            w[-2] = 1 / 3 + 0.5
            w[-1] = 0.5

        # this makes a tensor of size [1, 1, ..., size, ..., 1, 1]
        # this is so it will be broadcasted along the desired axis of weights.
        size = [1 for _ in range(dims)]
        size[dim] = grid_discretization_dimensions[dim]

        coords = torch.linspace(
            0, grid_spatial_dimensions[dim], grid_discretization_dimensions[dim]
        )
        delta = coords[1] - coords[0]

        weights *= delta * w.reshape(size)

    return weights


class Quadrature2d(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.cache = {}

        # putting this inside the class, instead of being global makes torch.compile not recompile
        self._quadrature_lookup = {
            REIMANN: reimann_quadrature_weights,
            TRAPEZOID: trapezoid_quadrature_weights,
            SIMPSON_13: simpsons_13_quadrature_weights,
        }

    def get_quadrature_weights(
        self,
        method: str,
        grid_spatial_dimensions: Tuple[int],
        grid_discretization_dimensions: Tuple[int],
        device: torch.device,
    ):
        return self._quadrature_lookup[method](
            grid_spatial_dimensions, grid_discretization_dimensions, device
        )

    def forward(self, x):
        if x.size() in self.cache:
            return self.cache[x.size()]
        # This is only used in subdomains, so can use (1, 1)
        quadrature_weights = self.get_quadrature_weights(
            self.method, (1, 1), (x.size(-2), x.size(-1)), device=x.device
        ).to(x.dtype)
        quadrature_weights.requires_grad_(False)
        self.cache[x.size()] = quadrature_weights
        return quadrature_weights


class Integral2d(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.quadrature = Quadrature2d(method)

    def forward(self, x):
        if self.method == REIMANN:
            x = x[..., :-1, :-1]
        quadrature_weights = self.quadrature(x)
        return (x * quadrature_weights).sum(dim=[-3, -2, -1])


class InnerProduct2d(Integral2d):
    def __init__(self, method):
        super().__init__(method)

    def forward(self, x, y):
        if self.method == REIMANN:
            x = x[..., :-1, :-1]
            y = y[..., :-1, :-1]
        quadrature_weights = self.quadrature(x)
        return einops.einsum(x * quadrature_weights, y, "... c h w -> ... c h w -> ...")
