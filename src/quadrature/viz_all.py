from omegaconf import OmegaConf
import hydra
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.simple_trainer import SimpleModule
from mondrian.dataset.allen_cahn import AllenCahnInMemoryDataset


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

font = {'size'   : 15}

matplotlib.rc('font', **font)


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    """
    This script visualizes the predictions of different models on the Allen-Cahn dataset in the appendix.
    """
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")
    model_paths = [
        'checkpoint1.ckpt',  
        'checkpoint2.ckpt',  
        'checkpoint3.ckpt',  
        ...
    ]
    model_names = [
        'LowRank',
        'Spec_Conv',
        'Interpolating',
        ...
        
    ]


    test_datasets = [
        AllenCahnInMemoryDataset(data_path)
        for data_path in cfg.experiment.test_data_paths
    ]
    fontsize = 12
    

    batch32 = test_datasets[0][2]
    input32, label32 = batch32
    label32 = label32.squeeze()
    
    
    batch128 = test_datasets[2][2]
    input128, label128 = batch128
    label128 = label128.squeeze()
    
    
    fig, axarr = plt.subplots(2*len(model_names)+2, 3, figsize=(7, 20), layout='constrained')
    fig.subplots_adjust(wspace=0.04, hspace=0.01)
    # input and label for 32x32 and 128x128
    im_input32 = axarr[0, 0].imshow(input32[0], cmap="plasma", vmin=-1, vmax=1)
    im_label32 = axarr[0, 1].imshow(label32, cmap="plasma", vmin=-1, vmax=1)
    im_input128 = axarr[1, 0].imshow(input128[0], cmap="plasma", vmin=-1, vmax=1)
    im_label128 = axarr[1, 1].imshow(label128, cmap="plasma", vmin=-1, vmax=1)

    fig.colorbar(im_input32, ax=axarr[0, 0], ticks=[-1, 0, 1], fraction=0.05)
    fig.colorbar(im_label32, ax=axarr[0, 1], ticks=[-1, 0, 1], fraction=0.05)
    fig.colorbar(im_input128, ax=axarr[1, 0], ticks=[-1, 0, 1], fraction=0.05)
    fig.colorbar(im_label128, ax=axarr[1, 1], ticks=[-1, 0, 1], fraction=0.05)
    axarr[0, 0].set_ylabel(r'$32\times 32$')
    axarr[1, 0].set_ylabel(r'$128\times 128$')
    axarr[0, 1].set_ylabel('Ground Truth', fontsize=fontsize, fontweight='bold')
    axarr[1, 1].set_ylabel('Ground Truth', fontsize=fontsize, fontweight='bold')
    # Plot input and label images
    for i, model_name in enumerate(model_names):
        module = SimpleModule.load_from_checkpoint(model_paths[i])
        module = module.cpu()
        with torch.no_grad():
            pred32 = module(input32.unsqueeze(0)).squeeze().detach()
            pred128 = module(input128.unsqueeze(0)).squeeze().detach()
        del module
        # Prediction plots
        im_pred = axarr[2*i+2, 1].imshow(pred32, cmap="plasma", vmin=-1, vmax=1)
        axarr[2*i+3, 1].imshow(pred128, cmap="plasma", vmin=-1, vmax=1)
        # Add a shared colorbar for both prediction plots
        fig.colorbar(im_pred, ax=axarr[2*i+2, 1], ticks=[-1, 0, 1], fraction=0.05)
        fig.colorbar(im_pred, ax=axarr[2*i+3, 1], ticks=[-1, 0, 1], fraction=0.05)
        axarr[2*i+2, 1].set_ylabel(model_name, fontsize=fontsize, fontweight='bold' )# name of the model
        
        # Error plots
        error32 = torch.abs(pred32 - label32)
        error128 = torch.abs(pred128 - label128)
        im_err32 = axarr[2*i+2, 2].imshow(error32, vmin=0, vmax=error32.max())
        fig.colorbar(im_err32, ax=axarr[2*i+2, 2], ticks=error32.max()*torch.tensor([0,0.5,1]), fraction=0.05, format="%4.0e")
        
        im_err128 = axarr[2*i+3, 2].imshow(error128, vmin=0, vmax=error128.max())
        fig.colorbar(im_err128, ax=axarr[2*i+3, 2], ticks=error128.max()*torch.tensor([0,0.5,1]), fraction=0.05, format="%4.0e")
        

        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

    for row in range(2*len(model_names)+2):
        fig.delaxes(axarr[row, 0]) if row not in [0,1] else fig.delaxes(axarr[row, 2])      
        
    plt.savefig('viz_all_ac.pdf',bbox_inches='tight')


if __name__ == "__main__":
    main()
