from omegaconf import OmegaConf
import hydra
import math

import torch

import lightning as L
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

from climate_learn.transforms import Denormalize

from mondrian.models import get_model
from mondrian.dataset.climate_learn.era5_dataloader import get_era5_dataloaders
from mondrian.trainer.era5_trainer import ERA5Module


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # need to set seeds for distributed training
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # use tensor cores
    torch.set_float32_matmul_precision("medium")
    # Enable TF32 on matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    # we use ddp with multiple GPUs, so the default learning
    # rate should be scaled linearly with the number of devices.
    device_count = torch.cuda.device_count()
    if device_count > 0:
        cfg.experiment.train_cfg.lr *= device_count
        cfg.experiment.train_cfg.warmup_iters *= math.sqrt(device_count)

    # build experiment dataloaders
    (dm, train_loader, val_loader, test_loader) = get_dataloaders(cfg)

    denormalize = Denormalize(dm)

    # Technically, this should be 141 in_channels (47 variables x 3 timestseps + 3 constants),
    # which is what they list in the paper,
    # but I think the loader repeats the constant fields... So it's just 49 x 3.
    # https://github.com/aditya-grover/climate-learn/blob/b48fb0242acc47e365af86bfbd9dd86e9dcbd6d2/src/climate_learn/data/iterdataset.py#L96C36-L96C49
    in_channels, out_channels = 147, 3

    # get model
    model = get_model(in_channels, out_channels, cfg.experiment.model_cfg)
    print(model)

    # setup lightning module
    max_steps = int(cfg.experiment.train_cfg.max_steps)
    module = ERA5Module(
        model,
        total_iters=max_steps,
        domain_size=cfg.experiment.model_cfg.domain_size,
        train_denormalize=denormalize,
        val_denormalize=denormalize,
        test_denormalize=denormalize,
        lr=cfg.experiment.train_cfg.lr,
        weight_decay=cfg.experiment.train_cfg.weight_decay,
        warmup_iters=cfg.experiment.train_cfg.warmup_iters,
    )

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, monitor="Val/MSELoss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="Val/MSELoss", min_delta=0.0, patience=10
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

    model_name = cfg.experiment.model_cfg.name
    lr = cfg.experiment.train_cfg.lr
    num_params = sum(p.numel() for p in model.parameters())
    logger_name = f"{model_name}_lr={lr}_params={num_params}"
    logger = WandbLogger(
        name=logger_name,
        project="mondrian_era5_5625",
        offline=cfg.wandb.offline,
    )

    # run training
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_steps=max_steps,
        gradient_clip_val=0.5,
    )
    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, test_loader)


def get_dataloaders(cfg):
    assert cfg.experiment.name == "era5"
    dm = get_era5_dataloaders(
        cfg.experiment.data_path,
        pred_range=cfg.experiment.train_cfg.pred_range,
        batch_size=cfg.experiment.train_cfg.batch_size,
        num_workers=cfg.experiment.train_workers,
        pin_memory=False,
    )
    return (
        dm,
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
    )


if __name__ == "__main__":
    main()
