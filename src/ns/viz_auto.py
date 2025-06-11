from omegaconf import OmegaConf
import hydra
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as ani

import seaborn

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.simple_trainer import SimpleModule
from mondrian.dataset.ns import PDEArenaNSDataset

from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

font = {'size' : 15}

matplotlib.rc('font', **font)


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    """
    This script visualizes the predictions of models on the Navier-Stokes dataset for main paper and appendix.
    """
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")

    model_paths = [
        'checkpoint1.ckpt',
        'checkpoint2.ckpt',
    ]
    model_names = [
        'ViT-NO',
        'Swin-NO',
    ]
    n_steps = 2
    roll_step = 2
    test_datasets = [
        PDEArenaNSDataset(data_path, False, roll_step * (n_steps+1))
        for data_path in [cfg.experiment.test_data_path]
    ]
    
    

    batch = test_datasets[0][0]
    input32, _ = batch
    input32 = torch.from_numpy(input32[:roll_step]) # get first timestep

    label_data = torch.from_numpy(batch[0])
    
    fig, axarr = plt.subplots(1 + 2 * len(model_paths), roll_step * (n_steps+1), constrained_layout=False, figsize=(7, 6))
    fig.subplots_adjust(wspace=0.04, hspace=0.01)
    vorticity_cmap = seaborn.color_palette('icefire', as_cmap=True)
    # Plot Ground Truth
    for i in range(roll_step*(n_steps+1)):
        im = axarr[0, i].imshow(label_data[i], cmap=vorticity_cmap)
    cbar = fig.colorbar(im, ax=axarr[0, :], fraction=0.04, shrink=0.7, pad=0.01)
    cbar.set_ticks([-3, 0, 3],)
    cbar.ax.tick_params(labelsize=10)
    fontsize = 8
    axarr[0, 0].set_ylabel('Ground Truth', fontsize=fontsize, fontweight='bold')

    # Plot Predictions and Errors
    max_error = 0 # For colorbar
    for model_idx, model_path in enumerate(model_paths):
        module = SimpleModule.load_from_checkpoint(model_path)
        module = module.cpu()
        rollout = [input32.unsqueeze(0)]
        # get autoregressive predictions
        for i in range(n_steps):
            with torch.no_grad():
                pred = module(rollout[-1]).detach()
            rollout.append(pred)
        del module
        rollout = torch.stack([x.squeeze().detach() for x in rollout]).flatten(0,1)
        error = torch.abs(label_data - rollout)
        
        # Plot Output
        for i in range(roll_step, roll_step*(n_steps+1)):
            im = axarr[2*model_idx+1, i].imshow(rollout[i], cmap=vorticity_cmap)
        cbar = fig.colorbar(im, ax=axarr[2*model_idx+1, :], fraction=0.04, shrink=0.7, pad=0.01)
        cbar.set_ticks([-3, 0, 3])
        cbar.ax.tick_params(labelsize=10)
        
        
        # Plot Error
        for i in range(roll_step, roll_step*(n_steps+1)):
            max_error = max(max_error, error.max())
            im = axarr[2*model_idx+2, i].imshow(error[i], vmin=0, vmax=max_error)
            
        cbar = fig.colorbar(im, ax=axarr[2*model_idx+2, :], fraction=0.04, shrink=0.7, pad=0.01, format='%.1f')
        cbar.set_ticks(max_error*torch.tensor([0, 0.5, 1]))
        cbar.ax.tick_params(labelsize=10) 


        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        
        
        axarr[2*model_idx+1, roll_step].set_ylabel(model_names[model_idx], fontsize=fontsize,fontweight='bold')
        axarr[2*model_idx+2, roll_step].set_ylabel(model_names[model_idx] + ' Error', fontsize=fontsize,fontweight='bold')

    for row in range(1, 2*len(model_paths)+1):
        for col in range(roll_step):
            fig.delaxes(axarr[row, col])

    plt.savefig('ns_comparison.pdf',bbox_inches='tight')
    plt.close()
    
    
if __name__ == "__main__":
    main()
