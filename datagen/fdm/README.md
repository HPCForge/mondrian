# FDM data generation.

These scripts use [py-pde](https://py-pde.readthedocs.io/en/latest/index.html) to solve PDEs using FDM.

Some utility functions may be defined in `mondrian_lib`

## Warning!

This uses a [cell-centered discretization](https://py-pde.readthedocs.io/en/latest/manual/mathematical_basics.html#spatial-discretization). Models like FNO like a uniform discretization, but the cell-centered discretization is not uniform due to the boundary conditions.
Thus, the data stored includes ghost-cells such that the boundary condition can be interpolated from the ghost-cell and interior cells.
