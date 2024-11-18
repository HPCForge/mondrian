from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        RichProgressBar
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger

from climate_learn.transforms import Denormalize

from mondrian.models import ViTOperator2d
from mondrian.dataset.climate_learn.era5_dataloader import (
    get_era5_dataloaders
)
from mondrian.trainer.era5_trainer import ERA5Module

@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    
    dtype = torch.bfloat16
    # use tensor cores
    torch.set_float32_matmul_precision('medium')
    # Enable TF32 on matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # build experiment dataloaders
    (dm, train_loader, val_loader) = get_dataloaders(cfg, dtype)
    
    denormalize = Denormalize(dm)

    # Technically, this should be 141 in_channels (47 variables x 3 timestseps + 3 constants),
    # but I think the loader repeats the constant fields... So it's just 49 x 3.
    # https://github.com/aditya-grover/climate-learn/blob/b48fb0242acc47e365af86bfbd9dd86e9dcbd6d2/src/climate_learn/data/iterdataset.py#L96C36-L96C49
    in_channels, out_channels = 147, 3

    # get model
    model = ViTOperator2d(in_channels, out_channels, 16, 4, 4, subdomain_size=(1, 1))

    # setup lightning module
    max_epochs = int(cfg.experiment.train_cfg.max_epochs)
    module = ERA5Module(
      model, 
      max_epochs, 
      train_denormalize=denormalize,
      val_denormalize=denormalize,
      test_denormalize=denormalize)

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor='Val/MSELoss',
            mode='min')
    early_stopping_callback = EarlyStopping(
            monitor='Val/MSELoss',
            min_delta=0.0,
            patience=5)
    progress_bar_callback = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )
    callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            progress_bar_callback
    ]
    
    logger = WandbLogger(project='mondrian_era5_5625', offline=cfg.wandb.offline)
    
    # run training
    trainer = L.Trainer(
        accelerator='gpu',
        # if device_count > 1, automatically uses ddp
        devices=torch.cuda.device_count(),
        logger=logger,
        callbacks=callbacks, 
        max_epochs=max_epochs,
        gradient_clip_val=0.5)
    trainer.fit(module, train_loader, val_loader)

def get_dataloaders(cfg, dtype):
    assert cfg.experiment.name == 'era5'    
    dm = get_era5_dataloaders(
        cfg.experiment.data_path,
        pred_range=24,
        batch_size=cfg.experiment.train_cfg.batch_size,
        num_workers=cfg.experiment.train_workers,
        pin_memory=False
    )
    return (
      dm,
      dm.train_dataloader(),
      dm.val_dataloader()      
    )

if __name__ == '__main__':
    main()
