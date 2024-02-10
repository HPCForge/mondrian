import numpy as np
import scipy
import scipy.sparse as sc
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch

def poisson_1d_matrix(n):
    ex = np.ones(n)
    data = np.array([-ex, 2 * ex, -ex])
    offsets = np.array([-1, 0, 1])
    T = sc.dia_array((data, offsets), shape=(n, n)).toarray()
    return T

@dataclass
class BoundaryCondition:
    top: np.array
    bottom: np.array
    left: np.array
    right: np.array

def write_boundary(u, g):
    u[-1, :] = g.top
    u[:, -1] = g.right
    u[0, :] = g.bottom
    u[:, 0] = g.left
    return u

def read_boundary(u):
    return BoundaryCondition(
        top=u[-1, :],
        right=u[:, -1],
        bottom=u[0, :],
        left=u[:, 0]
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
    """ Conctenate boundaries in clock-wise order
    """
    if torch.is_tensor(bc.top):
        return torch.cat((bc.top, bc.right, bc.bottom, bc.left))
    else:
        return np.concatenate((bc.top, bc.right, bc.bottom, bc.left))

def solve_poisson(
        g,
        f,
        xlim,
        ylim):
    r""" Solves the poisson equation on [0, xlim] x [0, ylim].
    div(grad(u)) = f, with boundary u = g.
    The discretization parameter, h, is determined from the resolution of f
    Args:
        g: 4xN vector, representing the boundary condition.
           this is unrolled around the domain.
        f: the forcing function.
        xlim: value of x on the right boundary
        ylim: values of y on the top boundary
    """
    # m is gridcells in x-direction, n is in y-direction
    m = g.top.shape[0]
    n = g.right.shape[0]

    # assume h is the same in x and y direction
    h = xlim / m

    x, y = np.meshgrid(np.linspace(0, xlim, m), np.linspace(0, ylim, n))
    assert x.max() == xlim
    assert y.max() == ylim

    is_bdy = np.full((n, m), True)
    is_bdy[1:-1, 1:-1] = False
    is_free = ~is_bdy.flatten()

    u = np.zeros((n, m))
    u = write_boundary(u, g)
    u = u.flatten()

    # we scale the rhs by h^2, rather than A
    b = np.zeros((n, m))
    b[1:-1,1:-1] = f
    b = b.flatten()
    b *= h**2

    # construct the sparse Poisson matrix
    Tx = poisson_1d_matrix(n)
    Ty = poisson_1d_matrix(m)
    A = sc.csr_array(sc.kron(sc.eye(n), Ty) + sc.kron(Tx, sc.eye(m)))
    assert A.shape[0] == m*n

    # shift the boundary condition to the rhs.
    b -= A @ u
    
    Ap = A[is_free,:][:,is_free]
    M = sc.lil_array(Ap.shape)
    M.setdiag(1/4)
    sol, info = sc.linalg.cg(A=Ap, b=b[is_free], M=M)
    assert info == 0
    u[is_free] = sol

    # u holds solution and the boundary
    u = u.reshape(n, m)
    return u

if __name__ == '__main__':
    xlim, ylim = 1, 1
    res_per_unit = 100
    m, n = xlim*res_per_unit, ylim*res_per_unit
    f = np.ones((m-2,n-2))
    g = BoundaryCondition(
        top=np.zeros(n), 
        right=np.ones(m),
        bottom=np.zeros(n),
        left=np.ones(m))
    u = solve_poisson(g, f, xlim, ylim)
    plt.imshow(u)
    plt.savefig('poisson_sol.png')
