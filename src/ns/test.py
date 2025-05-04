from omegaconf import OmegaConf
import hydra
import pathlib

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.simple_trainer import SimpleModule
from mondrian.dataset.pdearena_ns import PDEArenaNSDataset
from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator

@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")
    
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)

    test_loaders = get_test_loaders([cfg.experiment.test_data_path])

    module = SimpleModule.load_from_checkpoint(cfg.model_ckpt_path)
    print(module.model)

    trainer = L.Trainer(logger=False)

    for test_loader in test_loaders:
        trainer.test(module, dataloaders=test_loader)
        
    model = module.model
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{cfg.experiment.model_cfg.name} has {num_params} parameters')


def get_test_loaders(paths):
    print(paths)
    test_datasets = [PDEArenaNSDataset(data_path, transform=False) for data_path in paths]
    test_dataloaders = [DataLoader(dataset, batch_size=16) for dataset in test_datasets]
    return test_dataloaders

if __name__ == "__main__":
    main()
