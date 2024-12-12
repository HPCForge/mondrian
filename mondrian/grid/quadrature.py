from typing import Tuple

import torch
from torch import nn
import einops

from mondrian.constants import REIMANN, TRAPEZOID, SIMPSON_13


def integrate(f, quadrature_weights, dims):
    return (f * quadrature_weights).sum(dims)


def reimann_quadrature_weights(
    grid_spatial_dimensions: Tuple[int],
    grid_discretization_dimensions: Tuple[int],
):
    r"""
    Args:
      grid_spatial_dimensions: the phyiscal spatial dimensions of a domain.
                               I.e., [1, 1] would be a square domain.
      grid_discretization_dimensions: the number of points used in a discretization along each axis.
                                      I.e., [128, 64] is 128 points in the y direction and 64 in the x direction.
    Returns:
      The quadrature weights that can be used with a left or right Reimann sum.
    """
    # TODO: essentially, want x_2 - x_1 for the spacing.. I think 1 / J is not the same?
    deltas = torch.tensor(
        [
            grid_axis_spatial / (grid_axis_disc - 1)
            for (grid_axis_spatial, grid_axis_disc) in zip(
                grid_spatial_dimensions, grid_discretization_dimensions
            )
        ]
    )
    delta = torch.prod(deltas)

    # reimann sum excludes one of the endpoints.
    weight_sizes = [grid_axis_dim - 1 for grid_axis_dim in grid_discretization_dimensions]
    return torch.full(weight_sizes, delta)


def trapezoid_quadrature_weights(
    grid_spatial_dimensions: Tuple[int],
    grid_discretization_dimensions: Tuple[int],
    device=None,
):
    assert len(grid_spatial_dimensions) == len(grid_discretization_dimensions)
    dims = len(grid_spatial_dimensions)

    weight_sizes = [grid_axis_dim for grid_axis_dim in grid_discretization_dimensions]
    weights = torch.ones(weight_sizes, device=device)

    for dim in range(dims):

        w = torch.ones(grid_discretization_dimensions[dim], device=device)
        w[[0, -1]] = 0.5

        # this makes a tensor of size [1, 1, ..., size, ..., 1, 1]
        # this is so it will be broadcasted along the desired axis of weights.
        size = [1 for _ in range(dims)]
        size[dim] = grid_discretization_dimensions[dim]

        # TODO: double check this is right...
        delta = grid_spatial_dimensions[dim] / grid_discretization_dimensions[dim]

        weights *= delta * w.reshape(size)

    return weights


def simpsons_13_quadrature_weights(
    grid_spatial_dimensions: Tuple[int],
    grid_discretization_dimensions: Tuple[int],
    device=None,
):
    assert len(grid_spatial_dimensions) == len(grid_discretization_dimensions)
    dims = len(grid_spatial_dimensions)

    weight_sizes = [grid_axis_dim for grid_axis_dim in grid_discretization_dimensions]
    weights = torch.ones(weight_sizes, device=device)

    for dim in range(dims):

        w = torch.empty(grid_discretization_dimensions[dim], device=device)

        if grid_discretization_dimensions[dim] % 2 == 1:
            # if there is an odd number of grid-cells, we can use simpsons directly.
            # [1, 4, 2, 4, 2, ..., 2, 4, 1]
            w[[0, -1]] = 1
            w[1:-1:2] = 4
            w[2:-1:2] = 2
        else:
            # If there is an even number, we need to initialize first odd-part with simpsons,
            # and then have the last partition use the trapezoid rule.
            # [1, 4, 2, 4, 2, ...., 2, 4, 1.5, 0.5]
            w[[0, -2]] = 1
            w[1:-2:2] = 4
            w[2:-2:2] = 2
            w[-2] = 1.5
            w[-1] = 0.5

        # this makes a tensor of size [1, 1, ..., size, ..., 1, 1]
        # this is so it will be broadcasted along the desired axis of weights.
        size = [1 for _ in range(dims)]
        size[dim] = grid_discretization_dimensions[dim]

        # TODO: double check this is right...
        delta = grid_spatial_dimensions[dim] / grid_discretization_dimensions[dim]
        scale = delta / 3
        weights *= scale * w.reshape(size)

    return weights


_quadrature_lookup = {
    REIMANN: reimann_quadrature_weights,
    TRAPEZOID: trapezoid_quadrature_weights,
    SIMPSON_13: simpsons_13_quadrature_weights,
}


def get_quadrature_weights(
    method: str,
    grid_spatial_dimensions: Tuple[int],
    grid_discretization_dimensions: Tuple[int],
):
    return _quadrature_lookup[method](
        grid_spatial_dimensions, grid_discretization_dimensions
    )


class Quadrature2d(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.cache = {}

    def forward(self, x):
        if x.size() in self.cache:
            return self.cache[x.size()]
        quadrature_weights = get_quadrature_weights(
            self.method, (1, 1), (x.size(-2), x.size(-1))
        ).to(x)
        quadrature_weights.requires_grad_(False)
        self.cache[x.size()] = quadrature_weights
        return quadrature_weights
