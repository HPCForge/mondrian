from omegaconf import OmegaConf
import hydra
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.bubbleml_trainer import BubbleMLModule
from mondrian.dataset.bubbleml.bubbleml_forecast_dataset import BubbleMLForecastDataset
from mondrian.dataset.bubbleml.constants import unnormalize_data
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
        print(input[0:8].min(), input[0:8].max())
        print(input[8:16].min(), input[8:16].max())
        print(input[16:24].min(), input[16:24].max())
        print(input[24:32].min(), input[24:32].max())
        
        pred = module(input.unsqueeze(0), nuc.unsqueeze(0))
        pred = pred.detach()
        
        input = unnormalize_data(input.unsqueeze(0)).squeeze()
        pred = unnormalize_data(pred)
        label = unnormalize_data(label.unsqueeze(0)).squeeze()
                
        plot_results(input.squeeze(), pred.squeeze(), label.squeeze(), i)
        
def plot_results(input, output, label, i):
    fig, axarr = plt.subplots(5, 4, layout='constrained')
    
    print(output.size(), label.size())
    print(output.min(), output.max())

    print('temp')    
    print(output[16:24].min(), output[16:24].max())
    print(label[16:24].min(), label[16:24].max())
    
    vel_cmap = 'coolwarm'

    axarr[0, 0].imshow(torch.flipud(input[7]), vmin=-2, vmax=2, cmap=vel_cmap)
    axarr[0, 1].imshow(torch.flipud(input[15]), vmin=-2, vmax=2, cmap=vel_cmap)
    axarr[0, 2].imshow(torch.flipud(input[23]), vmin=50, vmax=95, cmap=temp_cmap())
    axarr[0, 3].imshow(torch.flipud(input[31])) 
    axarr[0, 0].set_ylabel('Input t')   

    axarr[1, 0].imshow(torch.flipud(output[0]), vmin=-2, vmax=2, cmap=vel_cmap)
    axarr[1, 1].imshow(torch.flipud(output[8]), vmin=-2, vmax=2, cmap=vel_cmap)
    axarr[1, 2].imshow(torch.flipud(output[16]), vmin=50, vmax=95, cmap=temp_cmap())
    axarr[1, 3].imshow(torch.flipud(output[24]), vmin=-0.5, vmax=0.5)
    axarr[1, 0].set_ylabel('Output t+1')
    
    axarr[2, 0].imshow(torch.flipud(output[7]), vmin=-2, vmax=2,  cmap=vel_cmap)
    axarr[2, 1].imshow(torch.flipud(output[15]), vmin=-2, vmax=2,  cmap=vel_cmap)
    axarr[2, 2].imshow(torch.flipud(output[23]), vmin=50, vmax=95, cmap=temp_cmap())
    axarr[2, 3].imshow(torch.flipud(output[31]))
    axarr[2, 0].set_ylabel('Output t+8')

    axarr[3, 0].imshow(torch.flipud(label[0]), vmin=-2, vmax=2, cmap=vel_cmap)
    axarr[3, 1].imshow(torch.flipud(label[8]), vmin=-2, vmax=2, cmap=vel_cmap)
    axarr[3, 2].imshow(torch.flipud(label[16]), vmin=50, vmax=95, cmap=temp_cmap())
    axarr[3, 3].imshow(torch.flipud(label[24]))
    axarr[3, 0].set_ylabel('Label t+1')
    
    axarr[4, 0].imshow(torch.flipud(label[7]), vmin=-2, vmax=2, cmap=vel_cmap)
    axarr[4, 1].imshow(torch.flipud(label[15]), vmin=-2, vmax=2, cmap=vel_cmap)
    axarr[4, 2].imshow(torch.flipud(label[23]), vmin=50, vmax=95, cmap=temp_cmap())
    axarr[4, 3].imshow(torch.flipud(label[31]))
    axarr[4, 0].set_ylabel('Label t+8')


    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.savefig(f"bubbleml_timestep_{i}.png")
    plt.close()    

if __name__ == "__main__":
    main()
