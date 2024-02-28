import pde
import numpy as np
import matplotlib.pyplot as plt
import h5py
from dataclasses import dataclass
from mondrian_lib.fdm_boundary_util import BoundaryCondition
from mondrian_lib.fdm_data_util import darcy_coeff
from typing import Callable
import numba as nb

class DampedKuramotoSivashinskyPDE(pde.PDEBase):
    r""" 
    A (possibly) damped form of the Kuramoto-Sivashinsky equation.
    nu controls the strength of the fourth order term
    alpha controls the strength of damping. alpha == 0 corresponds
    to the "normal" ks equation.

    This class is just a modificatin of
    https://github.com/zwicker-group/py-pde/blob/master/pde/pdes/kuramoto_sivashinsky.py
    to include the damping term
    """
    explicit_time_dependence = False

    def __init__(
        self,
        nu: float = 1,
        alpha: float = 0,
        *,
        bc = "auto_periodic_neumann",
        bc_lap = None,
    ):
        super().__init__()
        self.nu = nu
        self.alpha = alpha
        self.bc = bc
        self.bc_lap = bc if bc_lap is None else bc_lap

    def evolution_rate(  # type: ignore
        self,
        state: pde.ScalarField,
        t: float = 0,
    ) -> pde.ScalarField:
        assert isinstance(state, pde.ScalarField), "`state` must be ScalarField"
        state_lap = state.laplace(bc=self.bc, args={"t": t})
        result = (
            -self.nu * state_lap.laplace(bc=self.bc_lap, args={"t": t})
            - state_lap
            - self.alpha * state
            + state.gradient_squared(bc=self.bc, args={"t": t})
        )
        result.label = "evolution rate"
        return result

    def _make_pde_rhs_numba(
        self, state: pde.ScalarField
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        arr_type = nb.typeof(state.data)
        signature = arr_type(arr_type, nb.double)

        nu_value = self.nu
        alpha_value = self.alpha
        laplace = state.grid.make_operator("laplace", bc=self.bc)
        laplace2 = state.grid.make_operator("laplace", bc=self.bc_lap)
        gradient_sq = state.grid.make_operator("gradient_squared", bc=self.bc)

        @nb.njit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            result = -laplace(state_data, args={"t": t})
            # this is += because laplace2 is applied to result,
            # which is negative.
            result += nu_value * laplace2(result, args={"t": t})
            result -= alpha_value * state_data
            #result -= 0.5 * gradient_sq(state_data, args={"t": t})
            # Lots of people write it -0.5 * grad_sq, but damped people don't
            result += gradient_sq(state_data, args={"t": t})
            return result

        return pde_rhs

def solve_kuramoto_sivashinsky(xlim, ylim):
    assert isinstance(xlim, int)
    assert isinstance(ylim, int)
    rows = int(ylim * 32)
    cols = int(xlim * 32)
    # The unit-grid scales with the resolution. So that each cell has "length" 1
    grid = pde.UnitGrid((rows, cols))
    state = pde.ScalarField.random_uniform(grid)

    nu = 1
    alpha = 0.1
    eq = DampedKuramotoSivashinskyPDE(nu, alpha)
    storage = pde.MemoryStorage()
    tracker=['progress', storage.tracker(1)]
    result = eq.solve(state, t_range=50, dt=1e-3, scheme='rk', tracker=tracker)

    return nu, alpha, storage

dataset_size = 1
with h5py.File('./kuramoto_sivashinsky.hdf5', 'w') as f:
    zfill_cnt = len(str(dataset_size))
    domain_sizes = [8]
    for domain_size in domain_sizes:
        size_group = f.create_group(f'{domain_size}_{domain_size}')
        for i in range(dataset_size):
            xlim, ylim = domain_size, domain_size
            nu, alpha, storage = solve_kuramoto_sivashinsky(xlim, ylim)
            solution = np.stack([s.data for s in storage]) 

            for j in range(solution.shape[0]):
                sol = solution[j]
                print(sol.min(), sol.max())
                plt.imshow(sol - sol.mean(), cmap='turbo')
                jfill = str(j).zfill(3)
                plt.savefig(f'ks{jfill}.png')

            g = size_group.create_group(str(i).zfill(zfill_cnt))
            g.create_dataset('solution', data=solution)
            g.attrs['nu'] = nu
            g.attrs['alpha'] = alpha
            g.attrs['xlim'] = xlim
            g.attrs['ylim'] = ylim
