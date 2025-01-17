r"""
This makes images showing the original image, decomposition, and conversion to a sequence. 
"""
from omegaconf import OmegaConf
import hydra
import matplotlib.pyplot as plt
import einops
import torch

from mondrian.grid.decompose import decompose2d
from mondrian.dataset.allen_cahn import AllenCahnInMemoryDataset

def plt_decompose(img, tag):
    plt.imshow(img, vmin=-1, vmax=1, cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{tag}_original.png', bbox_inches='tight')
    plt.close()
    
    img = einops.rearrange(img, 'h w -> () () h w')
    n_sub_x, n_sub_y = 4, 4
    dec = decompose2d(img, n_sub_x, n_sub_y)
    
    print(dec.size())
    
    fig, axarr = plt.subplots(n_sub_y, n_sub_x)
    for row in range(n_sub_y):
        for col in range(n_sub_x):
            axarr[row][col].imshow(dec[0, row * n_sub_x + col, 0], vmin=-1, vmax=1, cmap='plasma')
            axarr[row][col].set_xticks([])
            axarr[row][col].set_yticks([])
            
    plt.savefig(f'{tag}_decompose.png', bbox_inches='tight')
    plt.close()

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    test_datasets = [
        AllenCahnInMemoryDataset(data_path)
        for data_path in cfg.experiment.test_data_paths
    ]
    sizes = [32, 64, 128]
    
    high_res_dataset = test_datasets[2]
    input, label = high_res_dataset[0]
    input = input[0]
    label = label[0]
    
    plt_decompose(input, 'input')
    plt_decompose(label, 'label')
    
if __name__ == "__main__":
    main()