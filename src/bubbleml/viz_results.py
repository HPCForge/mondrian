from omegaconf import OmegaConf
import hydra
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.bubbleml_trainer import BubbleMLModule
from mondrian.dataset.bubbleml.bubbleml_forecast_dataset import BubbleMLForecastDataset

import sys
sys.path.append("./")
from plot_utils import temp_cmap

@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # enable using tensor cores
    #torch.set_float32_matmul_precision("medium")
    # Enable TF32 on matmul and cudnn.
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


    module = BubbleMLModule.load_from_checkpoint(cfg.model_ckpt_path)
    module = module.cpu()
    
    print(cfg)

    dataset = BubbleMLForecastDataset(cfg.experiment.train_data_path,
                                      cfg.experiment.num_input_timesteps,
                                      cfg.experiment.input_step_size,
                                      cfg.experiment.lead_time)
    
    for i in [100, 200, 300]:
        batch = dataset[i]
        input, nuc, label = batch
        input = torch.from_numpy(input)
        nuc = torch.from_numpy(nuc)
        label = torch.from_numpy(label)
                
        pred = module(input.unsqueeze(0), nuc.unsqueeze(0))
        pred = pred.detach()
        
        plot_results(input.squeeze(), pred.squeeze(), label.squeeze(), i)
        
def plot_results(input, output, label, i):
    fig, axarr = plt.subplots(5, 4)
    
    print(output[4].min(), output[4].max())
    print(label[4].min(), label[4].max())
    
    axarr[0, 0].imshow(torch.flipud(input[0]), vmin=-1, vmax=1, cmap='turbo')
    axarr[0, 1].imshow(torch.flipud(input[8]), vmin=-1, vmax=1, cmap='turbo')
    axarr[0, 2].imshow(torch.flipud(input[16]), cmap=temp_cmap())
    axarr[0, 3].imshow(torch.flipud(input[24])) 
    axarr[0, 0].set_ylabel('Input')   

    axarr[1, 0].imshow(torch.flipud(output[0]), vmin=-1, vmax=1, cmap='turbo')
    axarr[1, 1].imshow(torch.flipud(output[2]), vmin=-1, vmax=1, cmap='turbo')
    axarr[1, 2].imshow(torch.flipud(output[4]), cmap=temp_cmap())
    axarr[1, 3].imshow(torch.flipud(output[6]))
    axarr[1, 0].set_ylabel('Output 25')
    
    axarr[2, 0].imshow(torch.flipud(output[1]), vmin=-1, vmax=1, cmap='turbo')
    axarr[2, 1].imshow(torch.flipud(output[3]), vmin=-1, vmax=1, cmap='turbo')
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
