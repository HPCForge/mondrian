from omegaconf import OmegaConf
import hydra
import time

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
from mondrian.dataset.toy_dataset import ToyInMemoryDataset
from mondrian.trainer.simple_trainer import SimpleModule
from mondrian.grid.quadrature import (
    reimann_quadrature_weights,
    trapezoid_quadrature_weights,
    simpsons_13_quadrature_weights
)
from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator, get_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator, get_default_feed_forward_operator
from mondrian.layers.spectral_conv import set_default_spectral_conv_modes
from mondrian.layers.linear_operator import LinearOperator2d
from mondrian.attention.func_self_attention import FuncSelfAttention
from mondrian.grid.decompose import decompose2d, recompose2d

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(1, 32, kernel_size=1)
        self.l2 = FuncSelfAttention(32, 4, 'channel', False)
        self.l3 = torch.nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x, n, m):
        x = self.l1(x)
        x = decompose2d(x, n, m)
        x = self.l2(x, n, m)
        x = recompose2d(x, n, m)
        x = self.l3(x)
        return x

@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # need to set seeds for distributed training
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    
    torch.set_float32_matmul_precision("high")
    
    dtype = torch.float32
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)
    set_default_spectral_conv_modes(cfg.experiment.spectral_conv_modes)

    # build experiment dataloaders
    train_loader, val_loader, in_channels, out_channels = get_dataloaders(cfg, dtype)

    # get model
    #model = Model()
    model = get_default_feed_forward_operator(1, 1, 64)
    model = get_model(in_channels, out_channels, cfg.experiment.model_cfg)
    print(model)

    # setup lightning module
    max_steps = int(cfg.experiment.train_cfg.max_steps)
    module = SimpleModule(
        model,
        total_iters=max_steps,
        domain_size=cfg.experiment.model_cfg.domain_size,
        lr=cfg.experiment.train_cfg.lr,
        weight_decay=cfg.experiment.train_cfg.weight_decay,
        warmup_iters=cfg.experiment.train_cfg.warmup_iters,
    )

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, monitor="Val/L2Error", mode="min"
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
    if "quadrature_method" in cfg.experiment:
        quadrature = cfg.experiment.quadrature_method
    else:
        quadrature = ""
    num_params = sum(p.numel() for p in model.parameters())
    logger_name = f"{model_name}_lr={lr}_params={num_params}_{quadrature}"
    logger = WandbLogger(
        name=logger_name,
        version=str(time.time()),
        project="quadrature_toy",
        offline=cfg.wandb.offline,
    )

    # run training
    trainer = L.Trainer(
        accelerator="gpu",
        logger=logger,
        callbacks=callbacks,
        max_steps=max_steps,
        gradient_clip_val=0.1,
    )
    trainer.fit(module, train_loader, val_loader)


def get_dataloaders(cfg, dtype):
    batch_size = cfg.experiment.train_cfg.batch_size

    train_workers = cfg.experiment.train_workers
    test_workers = cfg.experiment.test_workers

    # get experiment dataset
    train_dataset = ToyInMemoryDataset(cfg.experiment.train_path)
    val_dataset = ToyInMemoryDataset(cfg.experiment.val_path)

    in_channels = train_dataset.in_channels
    out_channels = train_dataset.out_channels

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
