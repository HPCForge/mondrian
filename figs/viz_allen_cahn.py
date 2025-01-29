from omegaconf import OmegaConf
import hydra
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from scipy.interpolate import interpn


import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.dataset.allen_cahn import AllenCahnInMemoryDataset

IMG_TYPE = 'pdf'

def interp(data, xcoords, ycoords, target_coords):
    interp_data = interpn((xcoords, ycoords), data, target_coords)
    return interp_data

def cell_centered_points(xlim, ylim, xres, yres):
    delta_x = xlim / xres
    delta_y = ylim / yres
    x_coords = delta_x * (np.arange(0, xres) + 0.5)
    y_coords = delta_y * (np.arange(0, yres) + 0.5)
    return x_coords, y_coords
    
def interp_high_res(high_res_image, downsample_res):
    yres = high_res_image.size(0)
    xres = high_res_image.size(1)
    x_coords, y_coords = cell_centered_points(1, 1, yres, xres)
    down_x_coords, down_y_coords = cell_centered_points(1, 1, downsample_res[0], downsample_res[1])
    target_coords = np.stack(np.meshgrid(down_x_coords, down_y_coords), axis=-1)
    low_res_image = interp(high_res_image.numpy(), x_coords, y_coords, target_coords)
    return low_res_image
    
def plt_res_decomp(high_res_image):
    img16 = interp_high_res(high_res_image, (8, 8))
    img32 = interp_high_res(high_res_image, (16, 16))
    img64 = interp_high_res(high_res_image, (32, 32))
    
    vmin = high_res_image.min()
    vmax = high_res_image.max()

    fig, axarr = plt.subplots(1, 4, constrained_layout=True)
    # Remove whitespace from around the image
    #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    
    axarr[0].imshow(img16, vmin=vmin, vmax=vmax, cmap='plasma')
    axarr[1].imshow(img32, vmin=vmin, vmax=vmax, cmap='plasma')
    axarr[2].imshow(img64, vmin=vmin, vmax=vmax, cmap='plasma')
    axarr[3].imshow(high_res_image.T, vmin=vmin, vmax=vmax, cmap='plasma')
    

    grid_spacings = (2, 4, 8, 32)
    img_heights = (8, 16, 32, 128)
    for i in range(len(axarr)):
        xspace = grid_spacings[i]
        yspace = grid_spacings[i]
        img_height = img_heights[i]
        for x in range(xspace, img_height, xspace):
            x -= 0.5
            axarr[i].plot([x, x], [-.5, img_height - .5], color='white', linewidth=1)
        
        for y in range(yspace, img_height, yspace):
            y -= 0.5
            axarr[i].plot([-.5, img_height - .5], [y, y], color='white', linewidth=1)
            
        axarr[i].set_xticks([])
        axarr[i].set_yticks([])

    plt.savefig(f'ac_res_decomp.{IMG_TYPE}', bbox_inches='tight')
    
@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    test_datasets = [
        AllenCahnInMemoryDataset(data_path)
        for data_path in cfg.experiment.test_data_paths
    ]
    sizes = [32, 64, 128]
    
    high_res_dataset = test_datasets[2]
    with torch.no_grad():
        for i in range(3):
            batch = high_res_dataset[i]
            input, label = batch
            plt_res_decomp(input[0]) 
    
    """
    with torch.no_grad():
        for size, dataset in zip(sizes, test_datasets):
            for i in range(3):
                batch = dataset[i]
                input, label = batch
                fig, axarr = plt.subplots(1, 2, figsize=(7, 2))
                axarr[0].imshow(input[0], cmap="plasma")
                axarr[1].imshow(label.squeeze(), cmap="plasma")
                
                axarr[0].set_xticks([])  
                axarr[0].set_yticks([])  
                
                axarr[1].set_xticks([])  
                axarr[1].set_yticks([])  

                plt.tight_layout()
                plt.savefig(f"ac_{size}.png", bbox_inches='tight')
                plt.close()

    with torch.no_grad():
        for size, dataset in zip(sizes, test_datasets):
            for i in range(3):
                batch = dataset[i]
                input, label = batch
                
                plt.imshow(input[0], cmap='plasma')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f'ac_input.{IMG_TYPE}')
                plt.close()
                
                plt.imshow(label[0], cmap='plasma')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f'ac_label.{IMG_TYPE}')
                plt.close()
 
                di = decompose2d(input.unsqueeze(0), 4, 4).squeeze()
                fig, axarr = plt.subplots(1, 16, figsize=(7, 2))
                for i in range(16):
                    axarr[i].imshow(di[i, 0], cmap='plasma')
                    axarr[i].set_xticks([])
                    axarr[i].set_yticks([])
                plt.tight_layout()
                plt.savefig(f'ac_input_dec.{IMG_TYPE}')
                plt.close()

                dl = decompose2d(label.unsqueeze(0), 4, 4).squeeze()
                fig, axarr = plt.subplots(1, 16, figsize=(7, 2))
                for i in range(16):
                    axarr[i].imshow(dl[i], cmap='plasma')
                    axarr[i].set_xticks([])
                    axarr[i].set_yticks([])
                plt.tight_layout()
                plt.savefig(f'ac_label_dec.{IMG_TYPE}')
                plt.close()
    """
    
if __name__ == "__main__":
    main()
