import functools
import logging
from typing import Tuple

import torch
from torch import nn
import einops

from mondrian.constants import REIMANN, TRAPEZOID, SIMPSON_13, QUADRATURE_METHODS
from .utility import cell_centered_grid

from scipy import integrate

# This can be overwritten to change the integration method,
# instead of passing a string around everywhere.
DEFAULT_QUADRATURE_METHOD = REIMANN

def trapezoid_interior_weights_1d(domain_size, num_points, device):
    delta = domain_size / num_points
    w = torch.full((num_points,), delta, device=device)
    w[[0, -1]] = 0.5 * delta
    return w

def simpson_interior_weights_1d(domain_size, num_points, device):
    w = torch.empty(num_points, device=device)
    if num_points % 2 == 1:
        # if there is an odd number of grid points, we can use simpsons directly.
        # [1, 4, 2, 4, 2, ..., 2, 4, 1] / 3
        w[[0, -1]] = 1 / 3
        w[1:-1:2] = 4 / 3
        w[2:-1:2] = 2 / 3
    else:
        # If there is an even number, we need to initialize first odd-part with simpsons,
        # and then have the last partition use the trapezoid rule.
        # [1 / 3, 4 / 3, 2 / 3, 4 / 3, 2 / 3, ...., 2 / 3, 4 / 3, (1 / 3 + .5), 0.5]
        w[0] = 1 / 3
        w[1:-2:2] = 4 / 3
        w[2:-2:2] = 2 / 3
        w[-2] = (1 / 3 + 0.5)
        w[-1] = 0.5
    delta = domain_size / num_points
    return delta * w
    
def apply_linear_boundary_weights(w, domain_size, num_points):
    delta = domain_size / num_points
    boundary_delta = delta / 2
    # fit line going to left boundary
    w[0] += (5 / 2) * (boundary_delta / 2)
    w[1] += -(1 / 2) * (boundary_delta / 2)
    # fit line going to right boundary
    w[-2] += -(1 / 2) * (boundary_delta / 2)
    w[-1] += (5 / 2) * (boundary_delta / 2)
    return w

def apply_linear_boundary_weights2(w, domain_size, num_points):
    delta = domain_size / num_points
    boundary_delta = delta / 2
    # fit curve going to left boundary
    w[0] += 3 * (boundary_delta / 2)
    w[1] += -(3 / 2) * (boundary_delta / 2)
    w[2] += (1 / 2) * (boundary_delta / 2)
    # fit curve going to right boundary
    w[-3] += (1 / 2) * (boundary_delta / 2)
    w[-2] += -(3 / 2) * (boundary_delta / 2)
    w[-1] += 3 * (boundary_delta / 2)
    return w

def set_default_quadrature_method(method: str):
    global DEFAULT_QUADRATURE_METHOD
    assert method in QUADRATURE_METHODS
    logging.info(f'changing DEFAULT_QUADRATURE_METHOD to {method}')
    DEFAULT_QUADRATURE_METHOD = method

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
    deltas = torch.tensor(
        [
            grid_axis_spatial / grid_axis_disc
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
    weights = torch.ones(weight_sizes, device=device)

    for dim in range(dims):
        w = trapezoid_interior_weights_1d(grid_spatial_dimensions[dim], grid_discretization_dimensions[dim], device=device)
        w = apply_linear_boundary_weights(w, grid_spatial_dimensions[dim], grid_discretization_dimensions[dim])
        # this makes a tensor of size [1, 1, ..., size, ..., 1, 1]
        # this is so it will be broadcasted along the desired axis of weights.
        size = [1 for _ in range(dims)]
        size[dim] = grid_discretization_dimensions[dim]
        weights *= w.reshape(size)

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
        w = simpson_interior_weights_1d(grid_spatial_dimensions[dim], grid_discretization_dimensions[dim], device=device)
        w = apply_linear_boundary_weights(w, grid_spatial_dimensions[dim], grid_discretization_dimensions[dim])
        # this makes a tensor of size [1, 1, ..., size, ..., 1, 1]
        # this is so it will be broadcasted along the desired axis of weights.
        size = [1 for _ in range(dims)]
        size[dim] = grid_discretization_dimensions[dim]

        weights *= w.reshape(size)

    return weights.to(torch.float32)

#@functools.lru_cache(maxsize=10)
def get_quadrature_weights(grid_spatial_dimensions: Tuple[int],
                           grid_discretization_dimensions: Tuple[int],
                           device: torch.device = None):
    assert DEFAULT_QUADRATURE_METHOD in QUADRATURE_METHODS
    if DEFAULT_QUADRATURE_METHOD == REIMANN:
        quad_func = reimann_quadrature_weights
    elif DEFAULT_QUADRATURE_METHOD == TRAPEZOID:
        quad_func = trapezoid_quadrature_weights
    elif DEFAULT_QUADRATURE_METHOD == SIMPSON_13:
        quad_func = simpsons_13_quadrature_weights
    quadrature_weights = quad_func(grid_spatial_dimensions, grid_discretization_dimensions, device=device)
    return quadrature_weights

#@functools.lru_cache(maxsize=10)
def get_unit_quadrature_weights(grid_discretization_dimensions: Tuple[int],
                                device: torch.device = None):
    assert DEFAULT_QUADRATURE_METHOD in QUADRATURE_METHODS    
    if DEFAULT_QUADRATURE_METHOD == REIMANN:
        quad_func = reimann_quadrature_weights
    elif DEFAULT_QUADRATURE_METHOD == TRAPEZOID:
        quad_func = trapezoid_quadrature_weights
    elif DEFAULT_QUADRATURE_METHOD == SIMPSON_13:
        quad_func = simpsons_13_quadrature_weights
    grid_spatial_dimensions = tuple([1 for _ in range(len(grid_discretization_dimensions))])
    quadrature_weights = quad_func(grid_spatial_dimensions, grid_discretization_dimensions, device=device)
    quadrature_weights.requires_grad_(False)
    return quadrature_weights