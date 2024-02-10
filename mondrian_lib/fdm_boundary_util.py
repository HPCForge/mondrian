"""
These are utilties for working with boundary conditions
in 2D finite difference problems. 
I try to keep functions generic between numpy and torch
"""

import numpy as np
from dataclasses import dataclass
import torch
from typing import Union

@dataclass
class BoundaryCondition:
    top: Union[np.array, torch.tensor]
    bottom: Union[np.array, torch.tensor]
    left: Union[np.array, torch.tensor]
    right: Union[np.array, torch.tensor]

def write_boundary(u, g):
    u[-1, :] = g.top
    u[:, -1] = g.right
    u[0, :] = g.bottom
    u[:, 0] = g.left
    return u

def read_boundary(u):
    r"""
    extract the boundary condition for problems that use a
    cell-edge stencil.
    """
    return BoundaryCondition(
        top=u[-1, :],
        right=u[:, -1],
        bottom=u[0, :],
        left=u[:, 0]
    )

def _linear_interp_pairs(edge):
    """
    perform a linear interpolation of each pair
    """
    return (edge[0] + edge[1]) / 2

def read_boundary_cell_centered(u_full):
    """
    Extract an (approximate) boundary for problems that
    used a cell-centered stencil.
    For an NxN solution, u_full should be (N+2)x(N+2) due
    to the presence of ghost cells outside the boundary.
    This uses a linear interpolation to get boundary values.
    """
    if torch.is_tensor(u_full):
        stack_func = torch.stack
    else:
        stack_func = np.stack
    # due to the ghost cells, we only want the interior [1:-1] for each edge
    top_edge = stack_func((u_full[-2,1:-1], u_full[-1,1:-1]))
    right_edge = stack_func((u_full[1:-1,-2], u_full[1:-1,-1]))
    bottom_edge = stack_func((u_full[1,1:-1], u_full[0,1:-1]))
    left_edge = stack_func((u_full[1:-1,1], u_full[1:-1,0]))
    top=_linear_interp_pairs(top_edge)
    return BoundaryCondition(
        top=_linear_interp_pairs(top_edge),
        right=_linear_interp_pairs(right_edge),
        bottom=_linear_interp_pairs(bottom_edge),
        left=_linear_interp_pairs(left_edge),
    )

def boundary_to_torch(g):
    return BoundaryCondition(
        top=torch.from_numpy(g.top),
        right=torch.from_numpy(g.right),
        bottom=torch.from_numpy(g.bottom),
        left=torch.from_numpy(g.left)
    )

def boundary_to_numpy(g):
    return BoundaryCondition(
        top=g.top.numpy(),
        right=g.right.numpy(),
        bottom=g.bottom.numpy(),
        left=g.left.numpy()
    )

def boundary_to_vec(bc):
    """ 
    Conctenate boundaries in clock-wise order
    """
    if torch.is_tensor(bc.top):
        return torch.cat((bc.top, bc.right, bc.bottom, bc.left))
    else:
        return np.concatenate((bc.top, bc.right, bc.bottom, bc.left))
