from omegaconf import DictConfig, OmegaConf
import hydra

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping
)

# TODO: tidy up data loading stuff
from torch_geometric.data import DataLoader as PyGDataLoader

from mondrian.models import ViTOperator2d
from mondrian.dataset.poseidon.base import (
    get_dataset as get_poseidon_dataset,
    POSEIDON_DATSETS
)
from mondrian.dataset.reno_shear_layer_dataset import ShearLayerDataset
from mondrian.trainer.poseidon_trainer import PoseidonModule
from mondrian.trainer.reno_trainer import RENOModule

@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    dtype = torch.float32
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # use tensor cores
    torch.set_float32_matmul_precision('medium')

    # get experiment dataset
    train_dataset, val_dataset = get_datasets(cfg, dtype)

    # build experiment dataloaders
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, cfg)

    # get model
    in_channels = train_dataset.in_channels
    out_channels = train_dataset.out_channels

    model = ViTOperator2d(in_channels, out_channels, 16, 4, 3, subdomain_size=(1, 1)).cuda()

    # setup lightning module
    max_epochs = int(cfg.experiment.train_cfg.max_epochs)
    total_iters = len(train_loader) * max_epochs
    module = get_module(cfg)(model, total_iters)

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
    callbacks = [
            checkpoint_callback,
            early_stopping_callback
    ]

    # run training
    trainer = L.Trainer(callbacks=callbacks, max_epochs=max_epochs)
    trainer.fit(module, train_loader, val_loader)

def get_module(cfg):
    name = cfg.experiment.name
    #if name == 'bubbleml':
    #    return BubbleMLModule
    if name in POSEIDON_DATSETS:
        return PoseidonModule
    elif name in ('shear_layer', 'disc_transport', 'poisson'):
        return RENOModule

def get_datasets(cfg, dtype):
    if cfg.experiment.name == 'bubbleml':
        # TODO: should make a train/val/test split, once I've generated a larger dataset
        #train_dataset = BubbleMLDataset(cfg.experiment.train_path, style='train', dtype=dtype)
        #test_dataset = BubbleMLDataset(cfg.experiment.test_path, style='test', dtype=dtype)
        pass
    if cfg.experiment.name in POSEIDON_DATSETS:
        kwargs = {
            'num_trajectories': 1000,
            'resolution': 64,
            # confused how to use these...
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
    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, cfg):
    batch_size = cfg.experiment.train_cfg.batch_size
    exp = ('bubbleml', 'shear_layer', 'disc_transport', 'poisson')

    # TODO: these should just be in experiment config
    train_workers = 2
    test_workers = 2
    # BubbleML test inputs are huge, so use more workers
    if exp == 'bubbleml':
        test_workers = 10

    if cfg.experiment.use_point:
        DL = PyGDataLoader
    else:
        DL = DataLoader
    
    #if cfg.experiment.name in exp:
    # TODO: add val loader when bubbleml ready
    train_loader = DL(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=train_workers,
                        pin_memory=True)
    val_loader = DL(val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=test_workers,
                    pin_memory=True)
    return train_loader, val_loader

if __name__ == '__main__':
    main()
