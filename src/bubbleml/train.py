from omegaconf import OmegaConf
import hydra
import time
import shutil
import math
import os
import pathlib

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBarTheme,
)
from lightning.pytorch.loggers import WandbLogger

from mondrian.dataset.bubbleml.bubbleml_forecast_dataset import BubbleMLForecastDataset
from mondrian.models import get_model
from mondrian.models.bubbleml_encoder import BubbleMLEncoder
from mondrian.trainer.bubbleml_trainer import BubbleMLModule
from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator
from mondrian.layers.spectral_conv import set_default_spectral_conv_modes


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # need to set seeds for distributed training
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    dtype = torch.float32

    # Enable tf32 tensor cores
    torch.set_float32_matmul_precision("high")
    
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)
    set_default_spectral_conv_modes(cfg.experiment.spectral_conv_modes)

    # build experiment dataloaders
    train_loader, val_loader, in_channels, out_channels = get_dataloaders(cfg, dtype)

    # Get model. All models use the BubbleMLEncoder to generically handle the nucleation sites.
    backbone_model = get_model(64, out_channels, cfg.experiment.model_cfg)
    model = BubbleMLEncoder(in_channels, out_channels=64, backbone_model=backbone_model)

    # setup lightning module
    max_epochs = int(cfg.experiment.train_cfg.max_epochs)
    max_steps = len(train_loader) * max_epochs
    # scale by sqrt to preserve effective variance with ddp training
    lr = math.sqrt(torch.cuda.device_count()) * cfg.experiment.train_cfg.lr
    warmup_iters = torch.cuda.device_count() * cfg.experiment.train_cfg.warmup_iters
    module = BubbleMLModule(
        model,
        total_iters=max_steps,
        domain_size=cfg.experiment.model_cfg.domain_size,
        lr=lr,
        weight_decay=cfg.experiment.train_cfg.weight_decay,
        warmup_iters=warmup_iters,
    )

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, monitor="Val/L2Error", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="Val/L2Error", min_delta=0.0, patience=25
    )
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
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        progress_bar_callback,
        lr_monitor_callback,
    ]

    # setup logger
    model_name = cfg.experiment.model_cfg.name
    lr = cfg.experiment.train_cfg.lr
    if "score_method" in cfg.experiment.model_cfg:
        quadrature = cfg.experiment.model_cfg.score_method
    else:
        quadrature = ""
    num_params = sum(p.numel() for p in model.parameters())
    logger_name = f"{model_name}_lr={lr}_params={num_params}_{quadrature}"
    logger = WandbLogger(
        name=logger_name,
        version=f'{model_name}_{num_params}_{time.time()}',
        project="bubbleml",
        offline=cfg.wandb.offline,
    )

    # run training
    trainer = L.Trainer(
        accelerator="gpu",
        logger=logger,
        callbacks=callbacks,
        max_steps=max_steps,
        gradient_clip_val=1.0,
    )
    trainer.fit(module, train_loader, val_loader)

def copy_to_scratch(filename):
    # TODO: this is very specific to hpc3...
    path = pathlib.Path(filename)
    name = path.name
    tmpdir = os.environ['TMPDIR']
    target_path = f'{tmpdir}/{name}'
    if not os.path.exists(target_path):
        shutil.copyfile(filename, target_path)
    return target_path

def get_dataloaders(cfg, dtype):
    batch_size = cfg.experiment.train_cfg.batch_size

    train_workers = cfg.experiment.train_workers
    test_workers = cfg.experiment.test_workers
    
    # copy data into node's scratch memory.
    #train_path = cfg.experiment.train_data_path
    #val_path = cfg.experiment.val_data_path
    train_path = copy_to_scratch(cfg.experiment.train_data_path)
    val_path = copy_to_scratch(cfg.experiment.val_data_path)

    # get experiment dataset
    train_dataset = BubbleMLForecastDataset(
        train_path, 
        cfg.experiment.num_input_timesteps, 
        cfg.experiment.input_step_size, 
        cfg.experiment.lead_time)
    
    val_dataset = BubbleMLForecastDataset(
        val_path, 
        cfg.experiment.num_input_timesteps, 
        cfg.experiment.input_step_size, 
        cfg.experiment.lead_time)

    in_channels = train_dataset.in_channels
    out_channels = train_dataset.out_channels

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, in_channels, out_channels


if __name__ == "__main__":
    main()
