import matplotlib
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

def temp_cmap():
    temp_ranges = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                    0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                   '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                   '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                   '#FC1407', '#FB0007']
    colors = list(zip(temp_ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
    return cmap

def vel_mag_cmap():
    samples = 16
    rainbow = matplotlib.colormaps['rainbow'].resampled(samples)
    vel_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                  0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_codes = [rainbow(i) for i in np.linspace(0, 1, len(vel_ranges))]
    colors = list(zip(vel_ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('velmag_colormap', colors)
    return cmap

def vel_cmap():
    return 'coolwarm'

def dfun_cmap():
    r"""
    This is a diverging colormap, centered at zero, so the bubble interface
    will be the divergent part. The cmap ranges from -12 to 1
    """
    cmap = matplotlib.colors.ListedColormap([
        matplotlib.colormaps['cividis'](0),
        matplotlib.colormaps['cividis'](256)
    ])
    return cmap