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

from mondrian_lib.data.bubbleml_dataset import BubbleMLDataset
from mondrian_lib.data.shear_layer_dataset import ShearLayerDataset
from mondrian_lib.data.disc_transport_dataset import DiscTransportDataset
from mondrian_lib.data.poisson_dataset import PoissonDataset
from mondrian_lib.data.point_dataset import PointDataset
from mondrian_lib.fdm.models.get_model import get_model
from mondrian_lib.trainer.bubbleml_trainer import BubbleMLModule
from mondrian_lib.trainer.reno_trainer import RENOModule
from mondrian_lib.trainer.reno_point_trainer import RENOPointModule
from mondrian_lib.dd_operator import DDNO
from mondrian_lib.dd_unet import DDGraphUNet

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

    #model = get_model(in_channels, out_channels, cfg.experiment.model_cfg, device)
    #model = DDNO(in_channels,
    #             out_channels,
    #             hidden_channels=32,
    #     )

    model = DDGraphUNet(in_channels=in_channels,
                        hidden_channels=64,
                        out_channels=out_channels,
                        subdomain_size=[(32, 32), (16, 16), (8, 8)])

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
    if name == 'bubbleml':
        return BubbleMLModule
    elif name in ('shear_layer', 'disc_transport', 'poisson'):
        if cfg.experiment.use_point:
            return RENOPointModule
        return RENOModule

def get_datasets(cfg, dtype):
    if cfg.experiment.name == 'bubbleml':
        # TODO: should make a train/val/test split, once I've generated a larger dataset
        train_dataset = BubbleMLDataset(cfg.experiment.train_path, style='train', dtype=dtype)
        test_dataset = BubbleMLDataset(cfg.experiment.test_path, style='test', dtype=dtype)
    elif cfg.experiment.name == 'shear_layer':
        train_dataset = ShearLayerDataset(cfg.experiment.data_path, which='training', s=64)
        val_dataset = ShearLayerDataset(cfg.experiment.data_path, which='validation', s=64)
        test_dataset = ShearLayerDataset(cfg.experiment.data_path, which='test', s=128)
    elif cfg.experiment.name == 'disc_transport':
        train_dataset = DiscTransportDataset(cfg.experiment.data_path, which='training')
        val_dataset = DiscTransportDataset(cfg.experiment.data_path, which='validation')
        test_dataset = DiscTransportDataset(cfg.experiment.data_path, which='test')
    elif cfg.experiment.name == 'poisson':
        train_dataset = PoissonDataset(cfg.experiment.data_path, which='training')
        val_dataset = PoissonDataset(cfg.experiment.data_path, which='validation')
        test_dataset = PoissonDataset(cfg.experiment.data_path, which='test')
    if cfg.experiment.use_point:
        train_dataset = PointDataset(train_dataset)
        val_dataset = PointDataset(val_dataset)
        test_dataset = PointDataset(test_dataset)
    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, cfg):
    batch_size = cfg.experiment.train_cfg.batch_size
    exp = ('bubbleml', 'shear_layer', 'disc_transport', 'poisson')

    train_workers = 2
    test_workers = 2
    # BubbleML test inputs are huge, so use more workers
    if exp == 'bubbleml':
        test_workers = 10

    if cfg.experiment.use_point:
        DL = PyGDataLoader
    else:
        DL = DataLoader
    
    if cfg.experiment.name in exp:
        # TODO: add val loader when bubbleml ready
        train_loader = DL(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=train_workers)
        val_loader = DL(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=test_workers)
    return train_loader, val_loader

if __name__ == '__main__':
    main()
