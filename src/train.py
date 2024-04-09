from omegaconf import DictConfig, OmegaConf
import hydra

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from mondrian_lib.data.bubbleml_dataset import BubbleMLDataset
from mondrian_lib.data.shear_layer_dataset import ShearLayerDataset
from mondrian_lib.fdm.models.get_model import get_model
from mondrian_lib.trainer.bubbleml_trainer import BubbleMLModule
from mondrian_lib.trainer.shear_layer_trainer import ShearLayerModule

@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    dtype = torch.float32
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # get experiment dataset
    train_dataset, test_dataset = get_datasets(cfg, dtype)

    # build experiment dataloaders
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, cfg)

    # get model
    in_channels = train_dataset.in_channels
    out_channels = train_dataset.out_channels
    model = get_model(in_channels, out_channels, cfg.experiment.model_cfg, device)

    # setup lightning module
    total_iters = len(train_loader) * int(cfg.experiment.train_cfg.max_epochs)
    module = ShearLayerModule(model, total_iters)

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor='Val/L2Error',
            mode='min')
    callbacks = [checkpoint_callback]

    # run training
    trainer = L.Trainer(callbacks=callbacks)
    trainer.fit(module, train_loader, test_loader)

def get_datasets(cfg, dtype):
    if cfg.experiment.name == 'bubbleml':
        # TODO: should make a train/val/test split, once I've generated a larger dataset
        train_dataset = BubbleMLDataset(cfg.experiment.train_path, style='train', dtype=dtype)
        test_dataset = BubbleMLDataset(cfg.experiment.test_path, style='test', dtype=dtype)
    elif cfg.experiment.name == 'shear_layer':
        train_dataset = ShearLayerDataset(cfg.experiment.data_path, which='training', s=64)
        val_dataset = ShearLayerDataset(cfg.experiment.data_path, which='validation', s=128)
        test_dataset = ShearLayerDataset(cfg.experiment.data_path, which='test', s=128)
    elif cfg.experiment.name == 'allen_cahn':
        pass
    return train_dataset, test_dataset

def get_dataloaders(train_dataset, test_dataset, cfg):
    batch_size = cfg.experiment.train_cfg.batch_size
    exp = ('bubbleml', 'shear_layer', 'allen_cahn')
    if cfg.experiment.name in exp:
        # TODO: add val loader when bubbleml ready
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    main()
