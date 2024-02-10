import numpy as np
from mondrian_lib import fdm_boundary_util
from mondrian_lib.fdm_boundary_util import BoundaryCondition


def test_constructor():
    bc = BoundaryCondition(top=1, right=1, bottom=1, left=1)
    assert bc.top == 1

def test_write_boundary():
    ones = np.ones(5)
    bc = BoundaryCondition(top=ones, right=ones, bottom=ones, left=ones)
    u = np.zeros((5, 5))
    u = fdm_boundary_util.write_boundary(u, bc)
    assert u[0,0] == 1
    assert (u[1:-1,1:-1] == 0).all()

def test_read_boundary():
    u = np.ones((5,5))
    bc = fdm_boundary_util.read_boundary(u)
    assert (bc.top == 1).all()

def test_read_boundary_cell_centered():
    u = np.ones((5,5))
    bc = fdm_boundary_util.read_boundary(u)
    assert (bc.top == 1).all()
    assert (bc.left == 1).all()
    assert (bc.right == 1).all()
    assert (bc.bottom == 1).all()


def test_read_boundary_cell_centered2():
    u = np.ones((7,7))
    u[0] = 2 # bottom ghost cells set to 2
    bc = fdm_boundary_util.read_boundary_cell_centered(u)
    # should only use interior cells
    assert bc.top.shape[0] == 5
    
    assert (bc.top == 1).all()
    assert (bc.left == 1).all()
    assert (bc.right == 1).all()
    assert (bc.bottom == 1.5).all()
