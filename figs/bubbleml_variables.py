import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.stats import skew

import sys
sys.path.append('./')

from plot_utils import temp_cmap, vel_cmap, dfun_cmap

IMG_TYPE = 'pdf'

def main():
    data_path = '/share/crsp/lab/amowli/share/mondrian/bubbleml/train.hdf5'
    
    with h5py.File(data_path, 'r') as handle:
        data = handle['Twall-92-40']
        timestep = 300
        
        fig, axarr = plt.subplots(1, 4, layout='constrained')
        im1 = axarr[0].imshow(np.flipud(data['velx'][timestep]), vmin=-2, vmax=2, cmap=vel_cmap())
        im2 = axarr[1].imshow(np.flipud(data['vely'][timestep]), vmin=-2, vmax=2, cmap=vel_cmap())
        im3 = axarr[2].imshow(np.flipud(data['temperature'][timestep]), vmin=50, vmax=95, cmap=temp_cmap())
        im4 = axarr[3].imshow(np.flipud(data['dfun'][timestep] > 0).astype(np.float32), cmap=dfun_cmap())
        
        cbar1 = fig.colorbar(im1, ax=axarr[0:2], location='top', shrink=0.5, pad=0.025, ticks=[-2, 0, 2])
        cbar3 = fig.colorbar(im3, ax=axarr[2], location='top', shrink=0.75, aspect=15, pad=0.025, ticks=[50, 75, 95])
        cbar4 = fig.colorbar(im4, ax=axarr[3], location='top', shrink=0.75, aspect=15, pad=0.025, ticks=[0, 1])
        
        cbar4.ax.set_xticklabels(['Liquid', 'Vapor'])
        
        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        axarr[0].set_xlabel('X Velocity')
        axarr[1].set_xlabel('Y Velocity')
        axarr[2].set_xlabel('Temperature')
        axarr[3].set_xlabel('Liquid-Vapor Mask')
                
        plt.savefig(f'bubbleml_variables.{IMG_TYPE}', bbox_inches='tight')
        plt.close()
       
if __name__ == "__main__":
    main()