from omegaconf import OmegaConf
import hydra
import pathlib

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.simple_trainer import SimpleModule
from mondrian.dataset.toy_dataset import ToyInMemoryDataset
from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator
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
        
    torch.set_float32_matmul_precision("high")
        
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)
    set_default_spectral_conv_modes(cfg.experiment.spectral_conv_modes)

    fine_loaders = get_test_loaders(cfg.experiment.fine_data_paths)
    test_loaders = get_test_loaders(cfg.experiment.test_data_paths)

    loaders = zip(fine_loaders, test_loaders)

    metrics = {}
    
    for path, (fine_loader, test_loader) in zip(cfg.experiment.test_data_paths, loaders):
        try:
            name = pathlib.Path(path).name
            print(name)
            module = SimpleModule.load_from_checkpoint(cfg.model_ckpt_path)
            trainer = L.Trainer(logger=False, max_epochs=3)
            #trainer.fit(module, train_dataloaders=fine_loader)
            m = trainer.test(module, dataloaders=test_loader)
            metrics[name] = m[0]['Test/L2Error']
            
            del trainer
            del module
        except:
            continue

    for k, v in metrics.items():
        print(f'{k} l2:  {v}')
      
def get_test_loaders(paths):
    test_datasets = [ToyInMemoryDataset(data_path) for data_path in paths]
    test_dataloaders = [DataLoader(dataset, batch_size=128) for dataset in test_datasets]
    return test_dataloaders


if __name__ == "__main__":
    main()
