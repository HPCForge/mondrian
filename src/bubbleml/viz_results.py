from omegaconf import OmegaConf
import hydra
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.bubbleml_trainer import BubbleMLModule
from mondrian.dataset.bubbleml.bubbleml_forecast_dataset import BubbleMLForecastDataset
from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator
from mondrian.layers.spectral_conv import set_default_spectral_conv_modes

import sys
sys.path.append("./")
from plot_utils import temp_cmap

@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("high")
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)
    set_default_spectral_conv_modes(cfg.experiment.spectral_conv_modes)

    module = BubbleMLModule.load_from_checkpoint(cfg.model_ckpt_path)
    module = module.cpu()
    
    print(cfg)

    dataset = BubbleMLForecastDataset(cfg.experiment.val_data_path,
                                      cfg.experiment.num_input_timesteps,
                                      cfg.experiment.input_step_size,
                                      cfg.experiment.lead_time)
    
    for i in [100, 200, 300]:
        batch = dataset[i]
        input, nuc, label = batch
        input = torch.from_numpy(input)
        nuc = torch.from_numpy(nuc)
        label = torch.from_numpy(label)
        
        print(input.size(), label.size())
        
        print(input[-1].min(), input[-1].max())
        print(input[0].min(), input[0].max())
        print(input[8].min(), input[8].max())
        print(input[16].min(), input[16].max())
        print(input[24].min(), input[24].max())
        #pred = module(input.unsqueeze(0), nuc.unsqueeze(0))
        #pred = pred.detach()
        
        #plot_results(input.squeeze(), pred.squeeze(), label.squeeze(), i)
        
def plot_results(input, output, label, i):
    fig, axarr = plt.subplots(5, 4)
    
    print(output.size(), label.size())
    
    print(output[6].min(), output[6].max())
    print(label[6].min(), label[6].max())
    
    axarr[0, 0].imshow(torch.flipud(input[0]), vmin=-1, vmax=1, cmap='turbo')
    axarr[0, 1].imshow(torch.flipud(input[8]), vmin=-1, vmax=1, cmap='turbo')
    axarr[0, 2].imshow(torch.flipud(input[16]), cmap=temp_cmap())
    axarr[0, 3].imshow(torch.flipud(input[24])) 
    #axarr[0, 4].imshow(torch.flipud(input[32])) 
    axarr[0, 0].set_ylabel('Input')   

    axarr[1, 0].imshow(torch.flipud(output[0]), vmin=-1, vmax=1,  cmap='turbo')
    axarr[1, 1].imshow(torch.flipud(output[2]), vmin=-1, vmax=1, cmap='turbo')
    axarr[1, 2].imshow(torch.flipud(output[4]), cmap=temp_cmap())
    axarr[1, 3].imshow(torch.flipud(output[6]))
    axarr[1, 0].set_ylabel('Output 25')
    
    axarr[2, 0].imshow(torch.flipud(output[1]), vmin=-1, vmax=1,  cmap='turbo')
    axarr[2, 1].imshow(torch.flipud(output[3]), vmin=-1, vmax=1,  cmap='turbo')
    axarr[2, 2].imshow(torch.flipud(output[5]), cmap=temp_cmap())
    axarr[2, 3].imshow(torch.flipud(output[7]))
    axarr[2, 0].set_ylabel('Output 50')

    axarr[3, 0].imshow(torch.flipud(label[0]), vmin=-1, vmax=1, cmap='turbo')
    axarr[3, 1].imshow(torch.flipud(label[2]), vmin=-1, vmax=1, cmap='turbo')
    axarr[3, 2].imshow(torch.flipud(label[4]), cmap=temp_cmap())
    axarr[3, 3].imshow(torch.flipud(label[6]))
    axarr[3, 0].set_ylabel('Label 50')
    
    axarr[4, 0].imshow(torch.flipud(label[1]), vmin=-1, vmax=1, cmap='turbo')
    axarr[4, 1].imshow(torch.flipud(label[3]), vmin=-1, vmax=1, cmap='turbo')
    axarr[4, 2].imshow(torch.flipud(label[5]), cmap=temp_cmap())
    axarr[4, 3].imshow(torch.flipud(label[7]))
    axarr[4, 0].set_ylabel('Label 50')


    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f"bubbleml_timestep_{i}.png")
    plt.close()    

if __name__ == "__main__":
    main()
