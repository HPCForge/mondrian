# FDM data generation.

These scripts use [py-pde](https://py-pde.readthedocs.io/en/latest/index.html) to solve simple PDEs using FDM.

# Environment Setup

This uses an annoying dependency (pylians) for generating the Gaussian random fields. 
I don't want it in the general environment. (It basically won't install
on mac unless you manually make changes to their setup.)
So I recommend making a separate environment if generating data, then just delete it once your done.

```console
micromamba env create -f env.yaml
```

# Note

pypde uses a [cell-centered discretization](https://py-pde.readthedocs.io/en/latest/manual/mathematical_basics.html#spatial-discretization). Models like FNO like a uniform discretization, but the cell-centered discretization is not uniform due to the boundary conditions. The data stored includes ghost-cells such that the boundary condition can be interpolated from the ghost-cell and interior cells. For most problems, these can just be ignored and the interior cells
can be used on their own.