# Solve Variable coefficient Poisson equation with dirichlet boundary
# 
# The problem formulation follows section 5.2 of "Larsson and Thomee, 2003"
# Implementation inspired by https://docu.ngsolve.org/latest/whetting_the_appetite/poisson.html
# 
# To use for training, this extract boundary and interior vertices,
# and gets the solution at the corresponding coordinates.

import argparse
import glob
import numpy as np
from ngsolve import *
import h5py
from pathlib import Path
import density_field_library as DFL

ngsglobals.msg_level = 1

def gaussian_field_3d(grid_res, k, Pk, seed):
    """ Use Pylians to generate a Gaussian random field. 
    Args:
        grid_res: number of grid cells
        k: 1D array of frequency
        Pk: 1D array of magnitudes to interpolate
        seed: Random seed
    Returns:
        A numpy array containing the field
    """
    # this is a somewhat arbitrary choice of box_size
    # it seems to spit out values in the range [-1, 1]
    box_size = 100.0 
    assert np.all(k >= 1)
    df_3d = DFL.gaussian_field_3D(grid_res, k, Pk, 0, seed,
                                  box_size, threads=1, verbose=True)
    print(df_3d.min(), df_3d.max())
    return df_3d

def random_gaussian_field_3d(grid_res, seed, highest_freq=None):
    """ Generates a random gaussian field interpolating
    power spectrum proportional to 1 / k**p.
    Args:
        grid_res: number of grid cells
        seed: a random seed controlling the power and gaussian field
    Returns:
        A VoxelCoefficient for a random Gaussian field.
    """
    if not highest_freq:
        highest_freq = grid_res / 2
    rng = np.random.default_rng(seed=seed)
    power = rng.uniform(low=2.0, high=3.0)
    k = 1 + np.arange(highest_freq).astype(np.float32)
    Pk = 1 / (k ** power).astype(np.float32)
    return gaussian_field_3d(grid_res, k, Pk, seed)

def get_rhs(grid_res, random, seed):
    if not random:
        return CoefficientFunction(32 * (z*(1-z) + y*(1-y) + x*(1-x)))
    else:
        df_3d = random_gaussian_field_3d(grid_res, seed)
        return VoxelCoefficient((0,0,0), (1,1,1), df_3d.astype(float), linear=True) 

def get_coeff(grid_res, min_coeff, random, seed):
    """ Coeff function should be smooth and positive.
    """
    if not random:
        return CoefficientFunction(0)
    else:
        df_3d = random_gaussian_field_3d(grid_res, seed)
        # force values to be greater than or equal to min_coeff
        df_3d += np.abs(df_3d.min()) + min_coeff
        return VoxelCoefficient((0,0,0), (1,1,1), df_3d.astype(float), linear=True) 

def get_boundary(grid_res, random, seed):
    """ Boundary function can have discontinuities.
    """
    if not random:
        return CoefficientFunction(0)
    else:
        df_3d = random_gaussian_field_3d(grid_res, seed)
        # force values to be greater than or equal to 0
        df_3d += np.abs(df_3d.min())
        print('BOUNDARY RANGE: ', df_3d.min(), df_3d.max())
        return VoxelCoefficient((0,0,0), (1,1,1), df_3d.astype(float), linear=True) 

def solve_poisson(mesh: Mesh,
                  grid_res: int,
                  random_rhs: bool,
                  random_coeff: bool,
                  random_boundary: bool,
                  min_coeff: float,
                  seed: int):
    assert min_coeff > 0
    assert isinstance(seed, int)

    # H1-conforming finite element space, all boundaries are dirichlet
    # TODO: this currently uses [1, ..., 6] to specify that each face of 
    # a cube is Dirichlet. This should be generalized to take arbitrary meshes.
    fes = H1(mesh, order=3, dirichlet=[1,2,3,4,5,6])
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # Get a few different seeds for each function
    # Pass around a seed instead of Generator because 
    # gaussian_field_3D takes a seed as input. 
    rng = np.random.default_rng(seed=seed)
    func_seeds = rng.integers(np.iinfo(np.int32).max, size=3, dtype=np.int32)

    # The right hand side
    f = LinearForm(fes)
    f += get_rhs(grid_res, random_rhs, func_seeds[0]) * v * dx

    coeff_func = get_coeff(grid_res, min_coeff, random_coeff, func_seeds[1])

    a = BilinearForm(fes, symmetric=True)
    a += coeff_func * grad(u) * grad(v) * dx

    a.Assemble()
    f.Assemble()

    # the solution field, with dirichlet boundary
    gfu = GridFunction(fes)
    boundary_func = get_boundary(grid_res, random_boundary, func_seeds[2]) 
    gfu.Set(boundary_func, BND)

    # solve Dirichlet problem, with the solution written to gfu
    pre = Preconditioner(a, 'local')
    solvers.BVP(bf=a, lf=f, gf=gfu, pre=pre)

    # convert forcing and coeffs to a grid function, so it can be queried
    forcing = GridFunction(fes)
    forcing.vec.data = f.vec

    return gfu, forcing, coeff_func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str, required=True)
    parser.add_argument('--write_dir', type=str, required=True)
    parser.add_argument('--problems_per_mesh', type=int, required=False, default=1)
    parser.add_argument('--constant_rhs', action='store_true', default=False)
    parser.add_argument('--constant_coeff', action='store_true', default=False)
    parser.add_argument('--constant_boundary', action='store_true', default=False,
                        help='''Generate random boundaries g. If False, solve
                                homogenous boundary problem''')
    args = parser.parse_args()

    mesh_files = glob.glob(f'{args.mesh_dir}/*.vol')

    # The grid resolution used by the gaussian random field.
    # This is just set to a resolution that should
    # be higher than the resolution of the domain.
    grf_grid_res = 128

    Path(args.write_dir).mkdir(parents=True, exist_ok=True)
    with h5py.File(f'{args.write_dir}/poisson.hdf5', 'w') as point_cloud_file:
        # solve multiple problems per mesh file, with different seeds
        repeat_mesh_files = [mesh_file for mesh_file in mesh_files for _ in range(args.problems_per_mesh)]
        for seed, mesh_file in enumerate(repeat_mesh_files):
            mesh = Mesh(mesh_file)
            sol_gfu, forcing_gfu, coeff_gfu = solve_poisson(
                    mesh, 
                    grf_grid_res,
                    not args.constant_rhs,
                    not args.constant_coeff,
                    not args.constant_boundary,
                    min_coeff=1,
                    seed=seed)

            # extract boundary vertex IDs from boundary elements
            boundary_vertices = set([])
            for el in mesh.Elements(BND):
                for v in el.vertices:
                    boundary_vertices.add(v)

            mesh_vertices = set([v for v in mesh.vertices])
            interior_vertices = mesh_vertices.difference(boundary_vertices)

            # extract boundary coordinates and values
            boundary_coords = np.empty((len(boundary_vertices), 3))
            boundary_values = np.empty((len(boundary_vertices), 1))
            for i, v in enumerate(boundary_vertices):
                p = mesh[v].point
                mp = mesh(p[0], p[1], p[2])
                for j in range(3):
                    boundary_coords[i, j] = p[j]
                boundary_values[i, 0] = sol_gfu(mp)

            # extract interior vertex coordinates, forcing function, and solution
            interior_coords = np.empty((len(interior_vertices), 3))
            interior_solution = np.empty((len(interior_vertices), 1))
            interior_forcing = np.empty((len(interior_vertices), 1))
            interior_coeff = np.empty((len(interior_vertices), 1))
            for i, v in enumerate(interior_vertices):
                p = mesh[v].point
                mp = mesh(p[0], p[1], p[2])
                for j in range(3):
                    interior_coords[i, j] = p[j]
                interior_solution[i, 0] = sol_gfu(mp)
                interior_forcing[i, 0] = forcing_gfu(mp)
                interior_coeff[i, 0] = coeff_gfu(mp)

            print(interior_solution)

            assert not np.isnan(interior_solution).any()
            assert not np.isnan(boundary_values).any()
            assert not np.isnan(interior_forcing).any()
            assert not np.isnan(interior_coeff).any()

            mesh_file_stem = str(seed).zfill(len(str(len(repeat_mesh_files))))
            point_cloud_group = point_cloud_file.create_group(mesh_file_stem)
            point_cloud_group.create_dataset('interior_coords', data=interior_coords)
            point_cloud_group.create_dataset('interior_solution', data=interior_solution)
            point_cloud_group.create_dataset('interior_forcing', data=interior_forcing)
            point_cloud_group.create_dataset('interior_coeff', data=interior_coeff)
            point_cloud_group.create_dataset('boundary_coords', data=boundary_coords)
            point_cloud_group.create_dataset('boundary_values', data=boundary_values)

if __name__ == '__main__':
    main()
