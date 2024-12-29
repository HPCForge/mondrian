import h5py
import matplotlib.pyplot as plt
import numpy as np


import sys
sys.path.append('./')

from plot_utils import temp_cmap

def main():
    data_path = '/pub/afeeney/Level-Set-Study/simulation/SubcooledBoiling/Train/train_fix_fix.hdf5'
    
    with h5py.File(data_path, 'r') as handle:
        print(handle.keys())
        temp_90 = handle['Twall-90-1']
        temp_95 = handle['Twall-95-54']
        temp_100 = handle['Twall-100-101']
        
        fig, axarr = plt.subplots(3, 5)
        plot_row_variables(axarr[0], temp_90, 100)
        plot_row_variables(axarr[1], temp_95, 100)
        plot_row_variables(axarr[2], temp_100, 100)
        
        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        axarr[0, 0].set_ylabel(r'$\degree 90$')
        axarr[1, 0].set_ylabel(r'$\degree 95$')
        axarr[2, 0].set_ylabel(r'$\degree 100$')
        
        axarr[2, 0].set_xlabel('X Velocity')
        axarr[2, 1].set_xlabel('Y Velocity')
        axarr[2, 2].set_xlabel('Temperature')
        axarr[2, 3].set_xlabel('Mass Flux')
        axarr[2, 4].set_xlabel('Distance Func')
                
        plt.tight_layout()
        plt.savefig('bubbleml_variables.png')
        plt.close()
        
        timesteps = [120, 140, 160, 180, 200]
        fig, axarr = plt.subplots(5, len(timesteps))
        plot_row_time(axarr[0], temp_95['velx'], timesteps)
        plot_row_time(axarr[1], temp_95['vely'], timesteps)
        plot_row_time(axarr[2], temp_95['temperature'], timesteps, vmin=50, vmax=100, cmap=temp_cmap())
        plot_row_time(axarr[3], temp_95['mass_flux'], timesteps)
        plot_row_time(axarr[4], temp_95['dfun'], timesteps)
        
        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        axarr[0, 0].set_ylabel('X Velocity')
        axarr[1, 0].set_ylabel('Y Velocity')
        axarr[2, 0].set_ylabel('Temperature')
        axarr[3, 0].set_ylabel('Mass Flux')
        axarr[4, 0].set_ylabel('Distance Func')
        
        axarr[4, 0].set_xlabel('Time ' + str(timesteps[0]))
        axarr[4, 1].set_xlabel(timesteps[1])
        axarr[4, 2].set_xlabel(timesteps[2])
        axarr[4, 3].set_xlabel(timesteps[3])
        axarr[4, 4].set_xlabel(timesteps[4])

        plt.tight_layout()
        plt.savefig('bubbleml_time.png')
        plt.close()

        2        
def plot_row_time(ax, data, timesteps, **kwargs):
    for idx, timestep in enumerate(timesteps):
        ax[idx].imshow(np.flipud(data[timestep]), **kwargs)
    

def plot_row_variables(ax, data, timestep):
    ax[0].imshow(np.flipud(data['velx'][timestep]))
    ax[1].imshow(np.flipud(data['vely'][timestep]))
    ax[2].imshow(np.flipud(data['temperature'][timestep]), vmin=50, vmax=100, cmap=temp_cmap())
    ax[3].imshow(np.flipud(data['mass_flux'][timestep]))
    ax[4].imshow(np.flipud(data['dfun'][timestep]))
    
if __name__ == "__main__":
    main()