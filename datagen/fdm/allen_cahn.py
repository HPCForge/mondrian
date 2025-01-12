import argparse
import pde
import numpy as np
import h5py
import multiprocessing
import time
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn

# This is from pylians: https://pylians3.readthedocs.io/en/master/gaussian_fields.html
import density_field_library as DFL


# This is from pylians: https://pylians3.readthedocs.io/en/master/gaussian_fields.html
import density_field_library as DFL


def gaussian_field_2d(grid_res, box_size, k, Pk, seed):
    """Use Pylians to generate a Gaussian random field.
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
    df_3d = DFL.gaussian_field_2D(grid_res, k, Pk, 0, seed, box_size, threads=1)
    assert np.isclose(df_3d.mean(), 0, atol=1e-3, rtol=1e-3)
    return df_3d


def allen_cahn_init(grid_res, box_size, power=3, pid=None):
    assert power > 0
    seed = int(time.time())
    if pid is not None:
        seed = seed ^ pid
    k = np.arange(1, grid_res + 1).astype(np.float32)
    pk = k**-power
    
    # truncate frequencies past 32...
    pk[32:] = 0
    
    grf = gaussian_field_2d(grid_res, box_size, k, pk, seed)
    grf = np.clip(grf, a_min=-0.9, a_max=0.9)
    
    # smooth a lot...
    grf = gaussian_filter(grf, sigma=5)
    return grf


def solve_allen_cahn(xlim, ylim, xres, yres, pid):
    assert isinstance(xlim, int)
    assert isinstance(ylim, int)
    grid = pde.CartesianGrid(([0, ylim], [0, xlim]), (yres, xres))

    field_dim = max(yres, xres)
    box_size = max(xlim, ylim)
    init = allen_cahn_init(field_dim, box_size, pid=pid)
    state_data = init[:yres, :xres]
    state = pde.ScalarField(grid, data=state_data)

    rng = np.random.default_rng(seed=int(time.time()) ^ pid)
    diffusivity = rng.uniform(low=5e-4, high=5e-3)

    end_time = 6
    storage = pde.MemoryStorage()
    tracker = [storage.tracker(end_time)]
    eq = pde.AllenCahnPDE(diffusivity, bc={"derivative": 0})
    eq.solve(state, t_range=end_time, dt=1e-4, adaptive=True, tracker=tracker, backend="numpy")

    return diffusivity, storage

def cell_centered_points(xlim, ylim, xres, yres):
    delta_x = xlim / xres
    delta_y = ylim / yres
    x_coords = delta_x * (np.arange(0, xres) + 0.5)
    y_coords = delta_y * (np.arange(0, yres) + 0.5)
    return x_coords, y_coords

def grid(xlim, ylim, xres, yres):
    xcoords = np.linspace(0, xlim, xres)
    ycoords = np.linspace(0, ylim, yres)
    return xcoords, ycoords
    
class Solver:
    def __init__(self, xlim, ylim, xres, yres):
        self.xlim = xlim
        self.ylim = ylim
        self.xres = xres
        self.yres = yres

    def __call__(self, pid):
        diffusivity, storage = solve_allen_cahn(
            self.xlim, self.ylim, self.xres, self.yres, pid
        )
        solution = np.stack([s.data for s in storage])
        return diffusivity, solution, self.xlim, self.ylim

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_res', required=True, type=int)
    parser.add_argument('--down_res', required=True, type=int)
    parser.add_argument('--num_sims', required=True, type=int)
    args = parser.parse_args()
    
    domain_size = (2, 2)
    resolution = (args.sim_res, args.sim_res)
    dataset_size = args.num_sims
    downsample_resolution = (args.down_res, args.down_res)
    
    # py-pde uses a cell-centered grid, but I interpolate to a vertex-based grid.
    x_coords, y_coords = cell_centered_points(1, 1, resolution[0], resolution[1])
    down_x_coords, down_y_coords = grid(1, 1, downsample_resolution[0], downsample_resolution[1])
    
    #cell_centered_points(1, 1, downsample_resolution[0], downsample_resolution[1])
    target_coords = np.stack(np.meshgrid(down_x_coords, down_y_coords), axis=-1)
    
    def interp(data):
        interp_data = interpn((x_coords, y_coords), data, target_coords, bounds_error=False, fill_value=None)
        return interp_data
    
    with h5py.File(f"./fix_grid_allen_cahn_{args.num_sims}_{args.down_res}.hdf5", "w") as f:
        zfill_cnt = len(str(dataset_size))
        xlim, ylim = domain_size
        xres, yres = resolution
        down_x, down_y = downsample_resolution
        size_group = f.create_group(f"res_{down_x}_{down_y}")
        processes = min(30, dataset_size)
        with multiprocessing.Pool(processes=processes) as pool:
            output = pool.map(Solver(xlim, ylim, xres, yres), range(dataset_size))
            
        for i, (diff, solution, xlim, ylim) in enumerate(output):
            print(solution.shape)
            print(interp(solution.transpose(1, 2, 0)).shape)
            g = size_group.create_group(str(i).zfill(zfill_cnt))
            g.create_dataset("solution", data=interp(solution.transpose(1, 2, 0)).transpose(2, 0, 1))
            g.attrs["diffusivity"] = diff
            g.attrs["xlim"] = xlim
            g.attrs["ylim"] = ylim
            g.attrs["xres"] = xres
            g.attrs["yres"] = yres