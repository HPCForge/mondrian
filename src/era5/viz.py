from omegaconf import DictConfig, OmegaConf
import hydra
import pathlib

import torch
from torch.utils.data import DataLoader
import lightning as L

from climate_learn.transforms import Denormalize

from mondrian.dataset.climate_learn.era5_dataloader import get_era5_dataloaders
from mondrian.dataset.climate_learn.visualize import visualize_at_index
from mondrian.trainer.era5_trainer import ERA5Module

@hydra.main(version_base=None, config_path='../../config', config_name='default')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # enable using tensor cores
    torch.set_float32_matmul_precision('medium')
    # Enable TF32 on matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # The visualization functions use dm.test_dataloader() internally
    dm = get_era5_dataloaders(
        cfg.experiment.data_path,
        pred_range=cfg.experiment.train_cfg.pred_range,
        batch_size=cfg.experiment.train_cfg.batch_size,
        num_workers=cfg.experiment.train_workers,
        pin_memory=False
    )
    
    ckpt_path = cfg.model_ckpt_path
    ckpt_dir = pathlib.Path(ckpt_path).parents[0]
    for out_var in dm.hparams.out_vars:
      (ckpt_dir / out_var).mkdir(parents=True, exist_ok=True)
    
    module = ERA5Module.load_from_checkpoint(ckpt_path)
    
    visualize_at_index(
      module,
      dm,
      in_transform=lambda x : x,
      out_transform=Denormalize(dm),
      out_variable='geopotential_500',
      variable_type='geopotential',
      save_path=ckpt_dir,
      src='era5',
      index=20
    )
    
    visualize_at_index(
      module,
      dm,
      in_transform=lambda x : x,
      out_transform=Denormalize(dm),
      out_variable='2m_temperature',
      variable_type='temperature',
      save_path=ckpt_dir,
      src='era5',
      index=20
    )

    visualize_at_index(
      module,
      dm,
      in_transform=lambda x : x,
      out_transform=Denormalize(dm),
      out_variable='temperature_850',
      variable_type='temperature',
      save_path=ckpt_dir,
      src='era5',
      index=20
    )

if __name__ == '__main__':
    main()