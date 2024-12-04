from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping
)

from climate_learn.transforms import Denormalize

from mondrian.models import ViTOperator2d, WinSAOperator2d
from mondrian.dataset.poseidon.base import (
    get_dataset as get_poseidon_dataset,
    POSEIDON_DATSETS
)
from mondrian.dataset.reno_shear_layer_dataset import ShearLayerDataset
from mondrian.trainer.poseidon_trainer import PoseidonModule
from mondrian.dataset.allen_cahn_dataset import AllenCahnDataset
from mondrian.trainer.reno_trainer import RENOModule
from pytorch_lightning import loggers as pl_loggers

@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    dtype = torch.float32
    # use tensor cores
    torch.set_float32_matmul_precision('medium')
    # Enable TF32 on matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # build experiment dataloaders
    train_loader, val_loader, in_channels, out_channels = get_dataloaders(cfg, dtype)

    # get model
    model = ViTOperator2d(in_channels, out_channels, 16, 4, 4, subdomain_size=(1, 1))

    # setup lightning module
    max_epochs = int(cfg.experiment.train_cfg.max_epochs)
    max_iters = len(train_loader) * max_epochs
    module = get_module(cfg)(model, max_iters)

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=False,
            monitor='Val/L2Error',
            mode='min')
    early_stopping_callback = EarlyStopping(
            monitor='Val/L2Error',
            min_delta=0.0,
            patience=25)
    callbacks = [
            checkpoint_callback,
            early_stopping_callback
    ]

    # run training

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = L.Trainer(callbacks=callbacks, max_epochs=max_epochs,logger=tb_logger)
    trainer.fit(module, train_loader, val_loader)

def get_module(cfg, **kwargs):
    name = cfg.experiment.name
    if name in POSEIDON_DATSETS:
        return PoseidonModule
    elif name in ('shear_layer', 'disc_transport', 'poisson', 'allen_cahn'):
        return RENOModule

def get_datasets(cfg, dtype):
    if cfg.experiment.name in POSEIDON_DATSETS:
        kwargs = {
            'num_trajectories': 1000,
            'resolution': 128,
            # TODO: confused how to use these...
            'max_num_time_steps': 4,
            'time_step_size': 4,
            'data_path': cfg.experiment.data_path
        }
        train_dataset = get_poseidon_dataset(
            cfg.experiment.name, which='train', **kwargs)
        val_dataset = get_poseidon_dataset(
            cfg.experiment.name, which='val', **kwargs)
        test_dataset = get_poseidon_dataset(
            cfg.experiment.name, which='test', **kwargs)
    elif cfg.experiment.name == 'shear_layer':
        train_dataset = ShearLayerDataset(cfg.experiment.data_path, which='training', s=128)
        val_dataset = ShearLayerDataset(cfg.experiment.data_path, which='validation', s=128)
        test_dataset = ShearLayerDataset(cfg.experiment.data_path, which='test', s=128)
    elif cfg.experiment.name == 'allen_cahn':
        train_dataset = AllenCahnDataset(cfg.experiment.data_path, which='training', in_steps=5, out_steps=5)
        val_dataset = AllenCahnDataset(cfg.experiment.data_path, which='validation', in_steps=5, out_steps=5)
        test_dataset = AllenCahnDataset(cfg.experiment.data_path, which='test', in_steps=5, out_steps=5)

    return train_dataset, val_dataset

def get_dataloaders(cfg, dtype):
    batch_size = cfg.experiment.train_cfg.batch_size
    
    train_workers = cfg.experiment.train_workers
    test_workers = cfg.experiment.test_workers
    
    # get experiment dataset
    train_dataset, val_dataset = get_datasets(cfg, dtype)
    in_channels = train_dataset.in_channels
    out_channels = train_dataset.out_channels
    
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

def get_model(model_cfg):
    if model_cfg.window:
        return WinSAOperator2d
    else:  
        return ViTOperator2d

if __name__ == '__main__':
    main()
