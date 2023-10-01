# Data Generation

There are several possible datasets that can be generated:
1. `laplace_box3d.sh` generates meshes in 3d boxes. Solves laplace equation with f=1, homogenous boundaries.

## Converting to hdf5

```console
python mfem_to_hdf5.py --srcdir ../../../neural-schwarz-data/box3d-mesh-sol/ --outdir ./data
```

## Dependencies:

1. [netgen](https://docu.ngsolve.org/latest/index.html) is used to generate finite element meshes.
2. [MFEM](https://mfem.org/) is used to solve different PDEs on the meshes.

MFEM should be installed manually. netgen is installed with the conda environment.

## Building

The build system for the mfem programs uses `make` and assumes a c++17 compiler.
You should specify the path to you mfem install. (This is the path to libmfem.a and mfem.hpp, after mfem is built.)

```console
make MFEM_PATH=</path/to/mfem/build/>
```

This produces binaries in the `build/` directory, that can be used
to generate different datasets.
