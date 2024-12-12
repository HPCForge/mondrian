import pde
import numpy as np
import h5py
import multiprocessing
import time
from scipy.ndimage import gaussian_filter

# This is from pylians: https://pylians3.readthedocs.io/en/master/gaussian_fields.html
import density_field_library as DFL

def gaussian_field_2d(grid_res, box_size, k, Pk, seed):
    """ Use Pylians to generate a Gaussian random field. 
    Args:
        grid_res: number of grid cells
        k: 1D array of frequency
        Pk: 1D array of magnitudes to interpolate. The problem should
            get "easier" as Pk goes to zero.
        seed: Random seed
    Returns:
        A numpy array containing the field
    """
    box_size = 10.0 * box_size
    assert np.all(k >= 0)
    df_3d = DFL.gaussian_field_2D(grid_res, 
                                  k, 
                                  Pk, 
                                  0, 
                                  seed,
                                  box_size,
                                  threads=1)
    assert np.isclose(df_3d.mean(), 0, atol=1e-3, rtol=1e-3)
    return df_3d

def allen_cahn_init(grid_res, box_size, power=3, pid=None):
    assert power > 0
    seed = int(time.time()) 
    if pid is not None:
        seed = seed ^ pid
    k = np.arange(1, grid_res+1).astype(np.float32)
    pk = k ** -power
    grf = gaussian_field_2d(grid_res, box_size, k, pk, seed)
    grf = np.clip(grf, a_min=-0.9, a_max=0.9)
    grf = gaussian_filter(grf, sigma=1)
    return grf

def solve_allen_cahn(xlim, ylim, xres, yres, pid):
    assert isinstance(xlim, int)
    assert isinstance(ylim, int)
    rows = int(ylim * yres)
    cols = int(xlim * xres)
    grid = pde.CartesianGrid(([0, ylim], [0, xlim]), (rows, cols))
    rng = np.random.default_rng()

    field_dim = max(rows, cols)
    box_size = max(xlim, ylim)
    init = allen_cahn_init(field_dim, box_size, pid=pid)
    state_data = init[:rows, :cols]
    state = pde.ScalarField(grid, data=state_data)

    diffusivity = rng.uniform(low=5e-4, high=5e-3)

    storage = pde.MemoryStorage()
    tracker=[storage.tracker(2)]
    eq = pde.AllenCahnPDE(diffusivity, bc={'derivative': 0})
    eq.solve(state, t_range=2, dt=1e-4, adaptive=True, tracker=tracker, backend='numpy')

    return diffusivity, storage

class Solver():
    def __init__(self, xlim, ylim, xres, yres):
        self.xlim = xlim
        self.ylim = ylim
        self.xres = xres
        self.yres = yres

    def __call__(self, pid):
        diffusivity, storage = solve_allen_cahn(self.xlim, 
                                                self.ylim,
                                                self.xres,
                                                self.yres,
                                                pid)
        solution = np.stack([s.data for s in storage]) 
        return diffusivity, solution, self.xlim, self.ylim

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    domain_sizes = [(1, 1)]
    resolutions = [(128, 128)]
    dataset_size = 1000
    with h5py.File('./allen_cahn.hdf5', 'w') as f:
        zfill_cnt = len(str(dataset_size))
        for domain_size in domain_sizes:
            for resolution in resolutions:
                print(f'{domain_size}')
                xlim, ylim = domain_size 
                xres, yres = resolution
                size_group = f.create_group(f'res_{xres}_{yres}')
                processes = min(30, dataset_size)
                with multiprocessing.Pool(processes=processes) as pool:
                    output = pool.map(Solver(xlim, ylim, xres, yres), range(dataset_size))
                for i, (diff, solution, xlim, ylim) in enumerate(output):
                    g = size_group.create_group(str(i).zfill(zfill_cnt))
                    g.create_dataset('solution', data=solution)
                    g.attrs['diffusivity'] = diff
                    g.attrs['xlim'] = xlim
                    g.attrs['ylim'] = ylim
                    g.attrs['xres'] = xres
                    g.attrs['yres'] = yres