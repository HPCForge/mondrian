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
    """
    This script visualizes the predictions of ViT and Swin models on the Allen-Cahn dataset in the main paper.
    """
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")
    module1 = SimpleModule.load_from_checkpoint('/checkpoint1.ckpt')# ViT-OP
    module2 = SimpleModule.load_from_checkpoint('/checkpoint2.ckpt')# SWIN-OP
    module1 = module1.cpu()
    module2 = module2.cpu()
    
    print(module1.model)
    print(module2.model)
    
    # sets the default quadrature method for any integrals evaluated by the model
    # set_default_quadrature_method(cfg.experiment.quadrature_method)
    # set_default_qkv_operator(cfg.experiment.linear_operator)
    # set_default_feed_forward_operator(cfg.experiment.neural_operator)

    test_datasets = [
        AllenCahnInMemoryDataset(data_path)
        for data_path in cfg.experiment.test_data_paths
    ]
    

    batch = test_datasets[0][2]
    input32, label32 = batch
    label32 = label32.squeeze()
    pred32_model1 = module1(input32.unsqueeze(0)).squeeze().detach()
    pred32_model2 = module2(input32.unsqueeze(0)).squeeze().detach()
    
    batch = test_datasets[2][2]
    input128, label128 = batch
    label128 = label128.squeeze()
    pred128_model1 = module1(input128.unsqueeze(0)).squeeze().detach()
    pred128_model2 = module2(input128.unsqueeze(0)).squeeze().detach()
    
    fig, axarr = plt.subplots(2, 6, figsize=(10, 3.2), layout='constrained')

    # Plot 32x32 results
    axarr[0, 0].imshow(input32[0], cmap="plasma", vmin=-1, vmax=1)
    axarr[0, 1].imshow(pred32_model1, cmap="plasma", vmin=-1, vmax=1)
    axarr[0, 2].imshow(pred32_model2, cmap="plasma", vmin=-1, vmax=1)
    im = axarr[0, 3].imshow(label32, cmap="plasma", vmin=-1, vmax=1)
    # Error bars for 32x32
    model1_error32 = torch.abs(pred32_model1 - label32)
    model2_error32 = torch.abs(pred32_model2 - label32)
    im = axarr[0, 4].imshow(model1_error32, vmin=0, vmax=max(model1_error32.max(), model2_error32.max()))
    im = axarr[0, 5].imshow(model2_error32, vmin=0, vmax=max(model1_error32.max(), model2_error32.max()))
    cbar = fig.colorbar(im, ax=axarr[0, 4:], fraction=0.05, format="%4.0e")
    cbar.set_ticks(max(model1_error32.max(), model2_error32.max()) * torch.tensor([0, 0.5, 1]))
    
    # plot 128x128 results
    axarr[1, 0].imshow(input128[0], cmap="plasma", vmin=-1, vmax=1)
    axarr[1, 1].imshow(pred128_model1, cmap="plasma", vmin=-1, vmax=1)
    axarr[1, 2].imshow(pred128_model2, cmap="plasma", vmin=-1, vmax=1)
    im = axarr[1, 3].imshow(label128, cmap="plasma", vmin=-1, vmax=1)
    fig.colorbar(im, ax=axarr[:, 3], ticks=[-1, 0, 1], fraction=0.05)
    
    # Error bars for 128x128
    model1_error128 = torch.abs(pred128_model1 - label128)
    model2_error128 = torch.abs(pred128_model2 - label128)
    im = axarr[1, 4].imshow(model1_error128, vmin=0, vmax=max(model1_error128.max(), model2_error128.max()))
    im = axarr[1, 5].imshow(model2_error128, vmin=0, vmax=max(model1_error128.max(), model2_error128.max()))
    cbar = fig.colorbar(im, ax=axarr[1, 4:], fraction=0.05, format="%4.0e")
    cbar.set_ticks(max(model1_error128.max(), model2_error128.max()) * torch.tensor([0, 0.5, 1]))

    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        
    axarr[0, 0].set_ylabel(r'$32\times 32$')
    axarr[1, 0].set_ylabel(r'$128\times 128$')
    axarr[1, 0].set_xlabel('Initial')
    axarr[1, 1].set_xlabel('ViT-NO')
    axarr[1, 2].set_xlabel('Swin-NO')
    axarr[1, 3].set_xlabel('Ground Truth')
    axarr[1, 4].set_xlabel('ViT Error')
    axarr[1, 5].set_xlabel('Swin Error')
    plt.savefig('ac_two_models_AbsError.pdf')


if __name__ == "__main__":
    main()
