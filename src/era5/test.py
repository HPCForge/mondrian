from omegaconf import DictConfig, OmegaConf
import hydra
import pathlib

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.dataset.climate_learn.era5_dataloader import get_era5_dataloaders
from mondrian.trainer.era5_trainer import ERA5Module

@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # enable using tensor cores
    torch.set_float32_matmul_precision('medium')
    # Enable TF32 on matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # build experiment dataloaders
    test_loader = get_test_dataloader(cfg)
    
    module = ERA5Module.load_from_checkpoint(cfg.model_ckpt_path)

    trainer = L.Trainer(logger=False)
    trainer.test(module, dataloaders=test_loader)
    
    # run a second time to save outputs
    accum = {'Input': [], 'Label': [], 'Pred': []} 
    for batch in test_loader:
        input = batch[0]
        label = batch[-1]
        pred = module(input).detach().cpu()
        accum['Input'].append(input)
        accum['Label'].append(label)
        accum['Pred'].append(pred)

    prefix = cfg.model_ckpt_path[:-len('.ckpt')]
    pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
    print(prefix)
    for k in accum:
        torch.save(torch.cat(accum[k], dim=0).detach().cpu(), f'{prefix}/{k}.pt')

def get_test_dataloader(cfg):
    assert cfg.experiment.name == 'era5'    
    dm = get_era5_dataloaders(
        cfg.experiment.data_path,
        pred_range=cfg.experiment.train_cfg.pred_range,
        batch_size=cfg.experiment.train_cfg.batch_size,
        num_workers=cfg.experiment.train_workers,
        pin_memory=False
    )
    return dm.test_dataloader()

if __name__ == '__main__':
    main()