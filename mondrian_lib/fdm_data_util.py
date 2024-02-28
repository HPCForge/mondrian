import numpy as np
import time
import density_field_library as DFL
from scipy.ndimage import gaussian_filter

def gaussian_field_2d(grid_res, box_size, k, Pk, seed):
    """ Use Pylians to generate a Gaussian random field. 
    Args:
        grid_res: number of grid cells
        k: 1D array of frequency
        Pk: 1D array of magnitudes to interpolate
        seed: Random seed
    Returns:
        A numpy array containing the field
    """
    box_size = 10.0 * box_size
    assert np.all(k >= 0)
    df_3d = DFL.gaussian_field_2D(grid_res, k, Pk, 0, seed,
                                  box_size, threads=1)
    assert np.isclose(df_3d.mean(), 0, atol=1e-3, rtol=1e-3)
    return df_3d

def darcy_coeff(grid_res, box_size, power=4, seed=None):
    assert power > 0
    if seed is None:
        seed = int(time.time())
    k = np.arange(1, grid_res+1).astype(np.float32)
    pk = k ** -power
    grf = gaussian_field_2d(grid_res, box_size, k, pk, seed)
    mask = grf <= 0
    grf[mask] = 3
    grf[~mask] = 12
    # FDM seems to have issues with discontinuous coefficients.
    grf = gaussian_filter(grf, sigma=1)
    return grf

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    p = darcy_coeff(512)
    plt.imshow(p)
    plt.savefig('darcy_coeff2.png')
    print('hi')
