from omegaconf import OmegaConf
import hydra

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        RichProgressBar,
        LearningRateMonitor
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger

from mondrian.models import get_model
from mondrian.dataset.allen_cahn import AllenCahnInMemoryDataset
from mondrian.trainer.simple_trainer import SimpleModule

@hydra.main(version_base=None, config_path='../../config', config_name='default')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    # need to set seeds for distributed training
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    
    dtype = torch.float32
    # use tensor cores
    torch.set_float32_matmul_precision('medium')
    # Enable TF32 on matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    
    # build experiment dataloaders
    train_loader, val_loader, in_channels, out_channels = get_dataloaders(cfg, dtype)

    # get model
    model = get_model(in_channels, out_channels, cfg.experiment.model_cfg)
    print(model)
    
    # setup lightning module
    max_steps = int(cfg.experiment.train_cfg.max_steps)
    module = SimpleModule(model, 
                          total_iters=max_steps,
                          domain_size=cfg.experiment.model_cfg.domain_size)

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor='Val/L2Error',
            mode='min')
    early_stopping_callback = EarlyStopping(
            monitor='Val/L2Error',
            min_delta=0.0,
            patience=25)
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
            metrics_format=".3e"))
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            progress_bar_callback,
            lr_monitor_callback
    ]

    # setup logger
    model_name = cfg.experiment.model_cfg.name
    lr = cfg.experiment.train_cfg.lr
    if 'score_method' in cfg.experiment.model_cfg:
      quadrature = cfg.experiment.model_cfg.score_method
    else:
      quadrature = ''
    num_params = sum(p.numel() for p in model.parameters())
    logger_name = f'{model_name}_lr={lr}_params={num_params}_{quadrature}'
    logger = WandbLogger(
            name=logger_name,
            version='vit_operator',
            project='quadrature_allen_cahn', 
            offline=cfg.wandb.offline)
    
    # run training
    trainer = L.Trainer(
        accelerator='gpu',
        logger=logger,
        callbacks=callbacks, 
        max_steps=max_steps,
        gradient_clip_val=0.5)
    trainer.fit(module, 
                train_loader, 
                val_loader)

def get_dataloaders(cfg, dtype):
    batch_size = cfg.experiment.train_cfg.batch_size
    
    train_workers = cfg.experiment.train_workers
    test_workers = cfg.experiment.test_workers
    
    # get experiment dataset
    dataset = AllenCahnInMemoryDataset(cfg.experiment.data_path)
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) - train_size)
    train_dataset, val_dataset = torch.utils.data.random_split(
      dataset, [train_size, val_size])
    in_channels = dataset.in_channels
    out_channels = dataset.out_channels
    
    #if cfg.experiment.name in exp:
    # TODO: add val loader when bubbleml ready
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=train_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=test_workers,
                            pin_memory=True)
    return train_loader, val_loader, in_channels, out_channels

if __name__ == '__main__':
    main()
