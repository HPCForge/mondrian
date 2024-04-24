r"""
Just a utility to visualize the fields as a sanity check
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

TEMPERATURE = 'temperature'
PRESSURE = 'pressure'
X_VELOCITY = 'velx'
Y_VELOCITY = 'vely'
DISTANCE_FUNC = 'dfun'
MASS_FLUX = 'mass_flux'

# Fields should be normalized and potentially passed into model
FIELDS = [
    TEMPERATURE,
    PRESSURE,
    X_VELOCITY,
    Y_VELOCITY,
    DISTANCE_FUNC,
    MASS_FLUX
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

with h5py.File('train.hdf5', 'r') as handle:
    grp = handle['Twall-100-101']

    for field in FIELDS:
        plt.imshow(np.flipud(grp[field][100]), vmin=-1, vmax=1, cmap='jet')
        plt.savefig(field)
        plt.close()

    print(grp['mass_flux'][100].min())
    print(grp['mass_flux'][100].max())
    print(grp['temperature'][100].min())
    print(grp['temperature'][100].max())
    print(grp['pressure'][107].max())
    print(grp['pressure'][107].max())

    #plt.imshow(np.flipud(np.log(grp['nucleation_dfun'][100])))
    #plt.savefig('nuc.png')
    #plt.close()

    plt.imshow(np.flipud(sigmoid(grp['dfun'][100])))
    plt.savefig('sig_dfun.png')
    plt.close()
