from omegaconf import OmegaConf
import hydra
import matplotlib.pyplot as plt

import numpy as np
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
from plot_utils import temp_cmap, vel_cmap, dfun_cmap

def denormalize_temp_grad(temp, t_wall, bulk_temp, thermal_conductivity):
    r"""
    """
    del_t = t_wall - bulk_temp
    return 2 * thermal_conductivity * del_t * (1 - temp)

def subcooled_heatflux(temp, dfun, t_wall, x, dy):
    r"""
    heat flux, q=dT/dy, is the temperature in the liquid phase in cells directly
    above the heater. 
    temp and dfun are layed out T x row x col
    dy is the grid spacing in the y direction.
    """
    assert temp.dim() == 3
    assert temp.size() == dfun.size()
    assert temp.size() == x.size()
    
    # These constants are specific to subcooled boiling.
    # Ideally, they should just be read from a simulation runtime parameters
    lc = 0.0007
    bulk_temp = 50
    thermal_conductivity = 0.054
    
    delta = 1 / 384
    x = delta * np.arange()
    
    d_temp = denormalize_temp_grad(temp[:, 0], t_wall, bulk_temp, thermal_conductivity)
    heater_mask = (x >= -2.5) & (x <= 2.5)
    liquid_mask = dfun < 0
    hflux_list = torch.mean((heater_mask[:, 0] & liquid_mask[:, 0]).to(float) * d_temp / (dy * lc),
                            dim=1)

    hflux = torch.mean(hflux_list)
    qmax = torch.max(hflux_list)
    return hflux, qmax

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
    module = module

    print(cfg)

    dataset = BubbleMLForecastDataset(cfg.experiment.test_data_path,
                                      cfg.experiment.num_input_timesteps,
                                      cfg.experiment.input_step_size,
                                      cfg.experiment.lead_time)

    with torch.no_grad():
        rollout_velx_pred = []
        rollout_vely_pred = []
        rollout_temp_pred = []
        rollout_dfun_pred = []

        rollout_velx_label = []
        rollout_vely_label = []
        rollout_temp_label = []
        rollout_dfun_label = []

        start_time = 100
        end_time = 300
        input, nuc, _ = dataset[start_time]
        input = torch.from_numpy(input).unsqueeze(0).cuda()
        nuc = torch.from_numpy(nuc).unsqueeze(0).cuda()

        rollout_velx_pred.append(unnormalize_data(input.detach().cpu().numpy())[:, 0:8])
        rollout_vely_pred.append(unnormalize_data(input.detach().cpu().numpy())[:, 8:16])
        rollout_temp_pred.append(unnormalize_data(input.detach().cpu().numpy())[:, 16:24])
        rollout_dfun_pred.append(unnormalize_data(input.detach().cpu().numpy())[:, 24:32])

        rollout_velx_label.append(unnormalize_data(input.detach().cpu().numpy())[:, 0:8])
        rollout_vely_label.append(unnormalize_data(input.detach().cpu().numpy())[:, 8:16])
        rollout_temp_label.append(unnormalize_data(input.detach().cpu().numpy())[:, 16:24])
        rollout_dfun_label.append(unnormalize_data(input.detach().cpu().numpy())[:, 24:32])

        for idx in range(start_time, end_time, cfg.experiment.lead_time):
            _, _, label = dataset[idx]
            label = torch.from_numpy(label).unsqueeze(0).cuda()
            input = module.forward(input, nuc)
            
            rollout_velx_pred.append(unnormalize_data(input.detach().cpu().numpy())[:, 0:8])
            rollout_vely_pred.append(unnormalize_data(input.detach().cpu().numpy())[:, 8:16])
            rollout_temp_pred.append(unnormalize_data(input.detach().cpu().numpy())[:, 16:24])
            rollout_dfun_pred.append(unnormalize_data(input.detach().cpu().numpy())[:, 24:32])

            rollout_velx_label.append(unnormalize_data(label.detach().cpu().numpy())[:, 0:8])
            rollout_vely_label.append(unnormalize_data(label.detach().cpu().numpy())[:, 8:16])
            rollout_temp_label.append(unnormalize_data(label.detach().cpu().numpy())[:, 16:24])
            rollout_dfun_label.append(unnormalize_data(label.detach().cpu().numpy())[:, 24:32])

        rollout_velx_pred = np.concatenate(rollout_velx_pred, axis=1).squeeze(0)
        rollout_vely_pred = np.concatenate(rollout_vely_pred, axis=1).squeeze(0)
        rollout_temp_pred = np.concatenate(rollout_temp_pred, axis=1).squeeze(0)    
        rollout_dfun_pred = np.concatenate(rollout_dfun_pred, axis=1).squeeze(0)

        rollout_velx_label = np.concatenate(rollout_velx_label, axis=1).squeeze(0)
        rollout_vely_label = np.concatenate(rollout_vely_label, axis=1).squeeze(0)
        rollout_temp_label = np.concatenate(rollout_temp_label, axis=1).squeeze(0)    
        rollout_dfun_label = np.concatenate(rollout_dfun_label, axis=1).squeeze(0)

    for idx in range(rollout_temp_pred.shape[0]):
        fig, axarr = plt.subplots(2, 4)
        axarr[0, 0].imshow(np.flipud(rollout_velx_pred[idx]), vmin=-2, vmax=2, cmap=vel_cmap())
        axarr[0, 1].imshow(np.flipud(rollout_vely_pred[idx]), vmin=-2, vmax=2, cmap=vel_cmap())
        axarr[0, 2].imshow(np.flipud(rollout_temp_pred[idx]), vmin=50, vmax=95, cmap=temp_cmap())
        axarr[0, 3].imshow(np.flipud(rollout_dfun_pred[idx]) > 0, cmap=dfun_cmap())

        axarr[1, 0].imshow(np.flipud(rollout_velx_label[idx]), vmin=-2, vmax=2, cmap=vel_cmap())
        axarr[1, 1].imshow(np.flipud(rollout_vely_label[idx]), vmin=-2, vmax=2, cmap=vel_cmap())
        axarr[1, 2].imshow(np.flipud(rollout_temp_label[idx]), vmin=50, vmax=95, cmap=temp_cmap())
        axarr[1, 3].imshow(np.flipud(rollout_dfun_label[idx]) > 0, cmap=dfun_cmap())

        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        idx = str(idx).zfill(4)
        plt.savefig(f'br/br_{idx}.png')
        plt.close()
    
if __name__ == "__main__":
    main()
