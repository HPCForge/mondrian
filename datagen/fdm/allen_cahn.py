import pde
import numpy as np
import matplotlib.pyplot as plt
import h5py
from dataclasses import dataclass
from mondrian_lib.fdm_boundary_util import BoundaryCondition
from mondrian_lib.fdm_data_util import darcy_coeff

def solve_allen_cahn(xlim, ylim):
    assert isinstance(xlim, int)
    assert isinstance(ylim, int)
    rows = int(ylim * 32)
    cols = int(xlim * 32)
    grid = pde.CartesianGrid(([0, ylim], [0, xlim]), (rows, cols))
    rng = np.random.default_rng()
    state_data = rng.uniform(low=0, high=1, size=(rows, cols))
    mask = state_data > 0.5
    state_data[mask] = 0.5
    state_data[~mask] = -0.5
    state = pde.ScalarField(grid, data=state_data)

    diffusivity = rng.uniform(low=1e-4, high=1e-3)

    storage = pde.MemoryStorage()
    tracker=[storage.tracker(0.1)]
    eq = pde.AllenCahnPDE(diffusivity, bc={'derivative': 0})
    eq.solve(state, t_range=6, dt=1e-4, adaptive=True, tracker=tracker, backend='numpy')

    return diffusivity, storage

max_lim = 9
domain_sizes = [
    (xlim, ylim)
    for ylim in range(2, max_lim)
    for xlim in range(ylim, max_lim)
]
dataset_size = 20 
with h5py.File('./allen_cahn.hdf5', 'w') as f:
    zfill_cnt = len(str(dataset_size))
    for domain_size in domain_sizes:
        print(f'{domain_size}')
        xlim, ylim = domain_size 
        size_group = f.create_group(f'{xlim}_{ylim}')
        for i in range(dataset_size):
            diffusivity, storage = solve_allen_cahn(xlim, ylim)
            solution = np.stack([s.data for s in storage]) 

            if abs(solution).max() > 1:
                print(f'Warning: {domain_size}, {i} did > 1')

            g = size_group.create_group(str(i).zfill(zfill_cnt))
            g.create_dataset('solution', data=solution)
            g.attrs['diffusivity'] = diffusivity
            g.attrs['xlim'] = xlim
            g.attrs['ylim'] = ylim
