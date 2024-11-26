from omegaconf import DictConfig, OmegaConf
import hydra
import pathlib

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.dataset.reno_shear_layer_dataset import ShearLayerDataset
from mondrian.dataset.allen_cahn_dataset import AllenCahnDataset
from mondrian.trainer.reno_trainer import RENOModule

@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # enable using tensor cores
    torch.set_float32_matmul_precision('medium')

    dtype = torch.float32
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # get experiment dataset
    test_dataset = get_datasets(cfg, dtype)

    # build experiment dataloaders
    test_loader = get_dataloaders(test_dataset, cfg)
    
    module = get_module(cfg).load_from_checkpoint(cfg.model_ckpt_path)
    trainer = L.Trainer(logger=False)
    trainer.test(module, dataloaders=test_loader)
    
    # run a second time to save outputs
    accum = {'Input': [], 'Label': [], 'Pred': []} 
    for i,batch in enumerate(test_loader):
        input = batch[0]
        label = batch[-1]
        pred = module(input).detach().cpu()
        accum['Input'].append(input)
        accum['Label'].append(label)
        accum['Pred'].append(pred)
        if i > 0:
            break
    prefix = cfg.model_ckpt_path[:-len('.ckpt')]
    pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
    print(prefix)
    for k in accum:
        torch.save(torch.cat(accum[k], dim=0).detach().cpu(), f'{prefix}/{k}.pt')

def get_module(cfg):
    #if cfg.experiment.name == 'bubbleml':
    #    return BubbleMLModule
    #else:
    return RENOModule

def get_datasets(cfg, dtype):
    #if cfg.experiment.name == 'bubbleml':
    #    test_dataset = BubbleMLDataset(cfg.experiment.test_path, style='test', dtype=dtype)
    #elif cfg.experiment.name == 'shear_layer':
    # test_dataset = ShearLayerDataset(cfg.experiment.data_path, which='test', s=128)
    #elif cfg.experiment.name == 'disc_transport':
    #    test_dataset = DiscTransportDataset(cfg.experiment.data_path, which='test')
    #elif cfg.experiment.name == 'poisson':
    #    test_dataset = PoissonDataset(cfg.experiment.data_path, which='test')

    test_dataset = AllenCahnDataset(cfg.experiment.data_path, which='test', in_steps=5, out_steps=5)
    return test_dataset

def get_dataloaders(test_dataset, cfg):
    batch_size = cfg.experiment.train_cfg.batch_size
    exp = ('shear_layer', 'disc_transport', 'poisson', 'allen_cahn')
    if cfg.experiment.name in exp:
        # TODO: add val loader when bubbleml ready
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif cfg.experiment.name == 'bubbleml':
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=10)
    return test_loader

if __name__ == '__main__':
    main()
