from mondrian_lib.fdm.cell_centered_coords import cell_centered_meshgrid
import torch

def test_cell_centered_meshgrid():
    xcoords, ycoords = cell_centered_meshgrid(1, 1, 20, 20)
    assert xcoords.min() > 0 and xcoords.max() < 1
    assert ycoords.min() > 0 and ycoords.max() < 1
    assert xcoords.size(0) == 20
    assert xcoords.size(1) == 20


def test_cell_centered_meshgrid2():
    xcoords, ycoords = cell_centered_meshgrid(1, 2, 20, 40)
    assert xcoords.min() > 0 and xcoords.max() < 1
    assert ycoords.min() > 0 and ycoords.max() < 2
    assert xcoords.size(0) == 40
    assert xcoords.size(1) == 20
    assert ycoords.size(0) == 40
    assert ycoords.size(1) == 20
    assert xcoords[0, 0] < xcoords[0, 1]
    assert ycoords[0, 0] < ycoords[1, 0]
    assert torch.isclose(torch.tensor(1 / 20),
                         xcoords[0, 1] - xcoords[0, 0])
