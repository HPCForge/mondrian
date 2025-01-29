import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

import sys
sys.path.append('./')

from plot_utils import temp_cmap, vel_mag_cmap

IMG_TYPE = 'pdf'

def main():
    data_path = '/share/crsp/lab/amowli/share/mondrian/bubbleml/train.hdf5'
    
    with h5py.File(data_path, 'r') as handle:
        print(handle.keys())
        temp_90 = handle['Twall-90-1']
        temp_92 = handle['Twall-92-40']
        temp_94 = handle['Twall-94-80']
        
        fig, axarr = plt.subplots(3, 4)
        plot_row_variables(axarr[0], temp_90, 100)
        plot_row_variables(axarr[1], temp_92, 100)
        plot_row_variables(axarr[2], temp_94, 100)

        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        axarr[0, 0].set_ylabel(r'Heater $\degree 90$')
        axarr[1, 0].set_ylabel(r'Heater $\degree 92$')
        axarr[2, 0].set_ylabel(r'Heater $\degree 94$')
        
        axarr[2, 0].set_xlabel('X Velocity')
        axarr[2, 1].set_xlabel('Y Velocity')
        axarr[2, 2].set_xlabel('Temperature')
        axarr[2, 3].set_xlabel('Distance Func')
                
        plt.tight_layout()
        plt.savefig(f'bubbleml_variables.{IMG_TYPE}', bbox_inches='tight')
        plt.close()
        
        timesteps = [120, 140, 160, 180, 200]
        fig, axarr = plt.subplots(4, len(timesteps), layout='constrained')
        vel_mag = np.sqrt(temp_92['velx'][:] ** 2 + temp_92['vely'][:] ** 2)
        plot_row_time(fig, axarr[0], temp_92['velx'], timesteps, vmin=-2, vmax=2, cmap='coolwarm')
        plot_row_time(fig, axarr[1], temp_92['vely'], timesteps, vmin=-2, vmax=2, cmap='coolwarm')
        plot_row_time(fig, axarr[2], temp_92['temperature'], timesteps, vmin=50, vmax=95, cmap=temp_cmap())
        plot_row_time(fig, axarr[3], (temp_92['dfun'][:] > 0).astype(np.float32), timesteps, vmin=0, vmax=1)
        
        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        axarr[0, 0].set_ylabel('X Velocity')
        axarr[1, 0].set_ylabel('Y Velocity')
        axarr[2, 0].set_ylabel('Temp.')
        axarr[3, 0].set_ylabel('Mask')
        
        axarr[3, 0].set_xlabel('Time ' + str(timesteps[0]))
        axarr[3, 1].set_xlabel(timesteps[1])
        axarr[3, 2].set_xlabel(timesteps[2])
        axarr[3, 3].set_xlabel(timesteps[3])
        axarr[3, 4].set_xlabel(timesteps[4])

        plt.savefig(f'bubbleml_time.{IMG_TYPE}', bbox_inches='tight')
        plt.close()
        
        fig, axarr = plt.subplots(3, 3)
        plot_histogram(axarr[0], temp_90)
        plot_histogram(axarr[1], temp_92)
        plot_histogram(axarr[2], temp_94)
        axarr[-1, 0].set_xlabel('X Velocity')
        axarr[-1, 1].set_xlabel('Y Velocity')
        axarr[-1, 2].set_xlabel('Temperature')
        axarr[0, 0].set_ylabel(r'Heater $\degree 90$')
        axarr[1, 0].set_ylabel(r'Heater $\degree 92$')
        axarr[2, 0].set_ylabel(r'Heater $\degree 94$')
        plt.tight_layout()
        plt.savefig(f'bubble_dist.{IMG_TYPE}', bbox_inches='tight')
        plt.close()

def plot_row_time(fig, ax, data, timesteps, **kwargs):
    for idx, timestep in enumerate(timesteps):
        im = ax[idx].imshow(np.flipud(data[timestep]), **kwargs)
    vmin = round(kwargs['vmin'], 1)
    vmax = round(kwargs['vmax'], 1)
    fig.colorbar(im, ax=ax, ticks=[vmin, vmax], pad=0.025)

def plot_row_variables(ax, data, timestep):
    ax[0].imshow(np.flipud(data['velx'][timestep]))
    ax[1].imshow(np.flipud(data['vely'][timestep]))
    ax[2].imshow(np.flipud(data['temperature'][timestep]), vmin=50, vmax=100, cmap=temp_cmap())
    ax[3].imshow(np.flipud(data['dfun'][timestep]))
    
def plot_histogram(ax, data):
    bins = 50
    range = np.linspace(-15, 15, bins)
    ax[0].bar(range, np.histogram(data['velx'][:], bins=bins, range=(-15, 15))[0])
    ax[1].bar(range, np.histogram(data['vely'][:], bins=bins, range=(-15, 15))[0])
    temp = data['temperature'][:]
    temp_hist = np.histogram(temp.astype(np.float64), bins=50)[0]
    ax[2].bar(np.linspace(temp.min(), temp.max(), 50), temp_hist)

if __name__ == "__main__":
    main()