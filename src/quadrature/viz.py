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

from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

font = {'size'   : 15}

matplotlib.rc('font', **font)


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")

    module = SimpleModule.load_from_checkpoint(cfg.model_ckpt_path)
    module = module.cpu()
    
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)

    test_datasets = [
        AllenCahnInMemoryDataset(data_path)
        for data_path in cfg.experiment.test_data_paths
    ]
    sizes = [32, 64, 128]
    

    batch = test_datasets[0][2]
    input32, label32 = batch
    label32 = label32.squeeze()
    pred32 = module(input32.unsqueeze(0)).squeeze().detach()
    
    batch = test_datasets[2][2]
    input128, label128 = batch
    label128 = label128.squeeze()
    pred128 = module(input128.unsqueeze(0)).squeeze().detach()
    
    fig, axarr = plt.subplots(2, 4, figsize=(7, 3.2), layout='constrained')

    axarr[0, 0].imshow(input32[0], cmap="plasma", vmin=-1, vmax=1)
    axarr[0, 1].imshow(pred32, cmap="plasma", vmin=-1, vmax=1)
    im = axarr[0, 2].imshow(label32, cmap="plasma", vmin=-1, vmax=1)
    #fig.colorbar(im, ax=axarr[0, 2], ticks=[-1, 0, 1], fraction=0.05)
    im = axarr[0, 3].imshow((pred32 - label32)**2)
    cbar = fig.colorbar(im, ax=axarr[0, 3], fraction=0.05, format="%4.0e")
    
    axarr[1, 0].imshow(input128[0], cmap="plasma", vmin=-1, vmax=1)
    axarr[1, 1].imshow(pred128, cmap="plasma", vmin=-1, vmax=1)
    im = axarr[1, 2].imshow(label128, cmap="plasma", vmin=-1, vmax=1)
    fig.colorbar(im, ax=axarr[:, 2], ticks=[-1, 0, 1], fraction=0.05)
    
    im = axarr[1, 3].imshow((pred128 - label128)**2)
    cbar = fig.colorbar(im, ax=axarr[1, 3], fraction=0.05, format="%4.0e")

    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        
    axarr[0, 0].set_ylabel(r'$32\times 32$')
    axarr[1, 0].set_ylabel(r'$128\times 128$')
    axarr[1, 0].set_xlabel('Initial')
    axarr[1, 1].set_xlabel('Prediction')
    axarr[1, 2].set_xlabel('Ground Truth')
    axarr[1, 3].set_xlabel('Squared Error')
    plt.savefig('ac.pdf')
    

    with torch.no_grad():
        for size, dataset in zip(sizes, test_datasets):
            for i in range(3):
                batch = dataset[i]
                input, label = batch
                label = label.squeeze()
                pred = module(input.unsqueeze(0)).squeeze().detach()
                
    fig, axarr = plt.subplots(1, 4, figsize=(7, 2), layout='constrained')
    
    axarr[0].imshow(input[0], cmap="plasma", vmin=-1, vmax=1)
    axarr[1].imshow(pred, cmap="plasma", vmin=-1, vmax=1)
    im = axarr[2].imshow(label, cmap="plasma", vmin=-1, vmax=1)
    fig.colorbar(im, ax=axarr[2], ticks=[-1, 0, 1], fraction=0.05)
    
    
    im = axarr[3].imshow((pred - label)**2)
    cbar = fig.colorbar(im, ax=axarr[3], fraction=0.05)
    cbar.formatter.set_powerlimits((0, 0))
    # to get 10^3 instead of 1e3
    cbar.formatter.set_useMathText(True)

    
    for i in range(len(axarr)):
        axarr[i].set_xticks([])
        axarr[i].set_yticks([])
    
    plt.savefig(f"ac_{size}.png")

if __name__ == "__main__":
    main()
