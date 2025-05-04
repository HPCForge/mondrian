from omegaconf import OmegaConf
import hydra
import os
import time
import pathlib
import shutil

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

from mondrian.models import get_model
from mondrian.dataset.ns import PDEArenaNSDataset
from mondrian.trainer.simple_trainer import SimpleModule
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

    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("medium")

    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)
    set_default_spectral_conv_modes(cfg.experiment.spectral_conv_modes)

    # build experiment dataloaders
    train_loader, val_loader, in_channels, out_channels = get_dataloaders(cfg, dtype)

    # get model
    model = get_model(in_channels, out_channels, cfg.experiment.model_cfg)

    # setup lightning module
    max_steps = int(cfg.experiment.train_cfg.max_steps)
    module = SimpleModule(
        model,
        total_iters=max_steps,
        domain_size=cfg.experiment.model_cfg.domain_size,
        lr=cfg.experiment.train_cfg.lr,
        weight_decay=cfg.experiment.train_cfg.weight_decay,
        warmup_iters=cfg.experiment.train_cfg.warmup_iters,
        eta_min=cfg.experiment.train_cfg.eta_min
    )

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, 
        save_last=True, 
        monitor="Val/L2Error", 
        mode="min",
        auto_insert_metric_name=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="Val/L2Error", min_delta=0.0, patience=50
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
    num_params = sum(p.numel() for p in model.parameters())
    slurm_job_id = os.environ['SLURM_JOB_ID']
    logger_name = f"{model_name}_lr={lr}_params={num_params}_jobid={slurm_job_id}_{time.time()}"
    logger = WandbLogger(
        # what appears on wandb website
        name=logger_name,
        # how it appears in local files
        version=logger_name,
        project="ns",
        offline=cfg.wandb.offline,
    )

    # run training
    trainer = L.Trainer(
        accelerator="gpu",
        logger=logger,
        callbacks=callbacks,
        max_steps=max_steps,
        gradient_clip_val=0.5,
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
    dataset = PDEArenaNSDataset(train_path, True)
    train_size = int(len(dataset) * 0.85)
    val_size = int(len(dataset) - train_size)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    val_dataset.transform = False
    in_channels = dataset.in_channels
    out_channels = dataset.out_channels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_workers,
    )
    return train_loader, val_loader, in_channels, out_channels


if __name__ == "__main__":
    main()
