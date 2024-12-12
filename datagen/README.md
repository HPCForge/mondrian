# Data Generation

We use data generated via finite difference (`fdm/`) and finite element (`fem/`) methods.

For FEM, the mesh generation and PDE solvers are decoupled. This
is to enable solving problems on a variety of meshes.

## Mesh Generation

The mesh generation code in `mesh` currently uses netgen.

## Solving PDEs

After generating meshes you can run solve a PDE on those meshes.
The `solve_netgen` directory has scripts using ngsolve. The `solve_mfem`
directory has some programs using mfem examples.