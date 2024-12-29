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

    module = BubbleMLModule.load_from_checkpoint(cfg.model_ckpt_path)
    module = module.cpu()

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
                
        pred = module(input.unsqueeze(0), nuc.unsqueeze(0))
        pred = pred.detach()
        
        plot_results(input.squeeze(), pred.squeeze(), label.squeeze(), i)
        
def plot_results(input, output, label, i):
    fig, axarr = plt.subplots(3, 4)
    
    axarr[0, 0].imshow(torch.flipud(input[0]), cmap='turbo')
    axarr[0, 1].imshow(torch.flipud(input[5]), cmap='turbo')
    axarr[0, 2].imshow(torch.flipud(input[10]), cmap=temp_cmap())
    axarr[0, 3].imshow(torch.flipud(input[15]))    
    
    axarr[1, 0].imshow(torch.flipud(output[0]), cmap='turbo')
    axarr[1, 1].imshow(torch.flipud(output[1]), cmap='turbo')
    axarr[1, 2].imshow(torch.flipud(output[2]), cmap=temp_cmap())
    axarr[1, 3].imshow(torch.flipud(output[3]))
    
    axarr[2, 0].imshow(torch.flipud(label[0]), cmap='turbo')
    axarr[2, 1].imshow(torch.flipud(label[1]), cmap='turbo')
    axarr[2, 2].imshow(torch.flipud(label[2]), cmap=temp_cmap())
    axarr[2, 3].imshow(torch.flipud(label[3]))
    
    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f"ac_{i}.png")
    plt.close()    

if __name__ == "__main__":
    main()
