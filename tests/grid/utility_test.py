from mondrian.grid.utility import grid


def test_grid():
    g = grid((128, 128), [4, 4])
    assert g.size(0) == 2
    assert g.size(1) == 128
    assert g.size(2) == 128
    assert g[0][0][0] == -2
