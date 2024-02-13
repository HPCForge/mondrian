import pde
import numpy as np
import matplotlib.pyplot as plt
import h5py
from dataclasses import dataclass
from mondrian_lib.fdm_boundary_util import BoundaryCondition
from mondrian_lib.fdm_data_util import darcy_coeff

class VariableDiffusionPDE(pde.PDEBase):
    def __init__(self, diffusivity: np.array, bc, forcing):
        self.diffusivity = diffusivity
        self.bc = bc
        self.forcing = forcing

    def evolution_rate(self, state, t=0):
        assert isinstance(state, pde.ScalarField)
        # supposedly more stable to expand the divergence:
        # div(k * grad(u)) == k*lap(u) + grad(k).grad(u)
        state_lap = self.diffusivity * state.laplace(bc=self.bc, args={'t': t})
        state_grad = state.gradient(bc=self.bc, args={'t', t})
        diff_grad = self.diffusivity.gradient(bc=self.bc)
        result = (
            state_lap
            + state_grad @ diff_grad
            + self.forcing
        )
        result.label = 'evolution rate'
        return result

def solve_diffusion(xlim, ylim):
    assert isinstance(xlim, int)
    assert isinstance(ylim, int)
    rows = int(ylim * 64)
    cols = int(xlim * 64)
    grid = pde.CartesianGrid(([0, ylim], [0, xlim]), (rows, cols))
    state = pde.ScalarField.random_uniform(grid)

    field_dim = max(rows, cols)
    box_size = max(xlim, ylim)
    coeff = darcy_coeff(field_dim, box_size)
    diffusivity = coeff[:rows, :cols]
    diffusivity = pde.ScalarField(grid, data=diffusivity)

    bc = {'value': 0}
    eq = VariableDiffusionPDE(diffusivity, bc=bc, forcing=1)

    storage = pde.MemoryStorage()
    solver = pde.ScipySolver(eq)
    controller = pde.Controller(solver,
                                t_range=0.02,
                                tracker=['progress', storage.tracker(0.001)])
    result = controller.run(state, dt=1e-4)

    return diffusivity, storage

dataset_size = 10
with h5py.File('./diffusion.hdf5', 'w') as f:
    zfill_cnt = len(str(dataset_size))
    domain_sizes = [1, 2, 4]
    for domain_size in domain_sizes:
        size_group = f.create_group(f'{domain_size}_{domain_size}')
        for i in range(dataset_size):
            xlim, ylim = domain_size, domain_size
            diffusivity, storage = solve_diffusion(xlim, ylim)
            diff = np.squeeze(diffusivity.data)
            solution = np.stack([s._data_full for s in storage]) 
            g = size_group.create_group(str(i).zfill(zfill_cnt))
            g.create_dataset('diffusivity', data=diff)
            g.create_dataset('solution', data=solution)
            g.attrs['xlim'] = xlim
            g.attrs['ylim'] = ylim

#for i in range(len(storage)):
#    fix, axarr = plt.subplots(1, 2)
#    axarr[0].imshow(diff.data)
#    axarr[1].imshow(storage[i].data)
#    plt.savefig(f'result{i}.png')
