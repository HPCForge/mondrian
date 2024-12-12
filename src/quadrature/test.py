from omegaconf import OmegaConf
import hydra
import pathlib

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.simple_trainer import SimpleModule
from mondrian.dataset.allen_cahn import AllenCahnInMemoryDataset


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # enable using tensor cores
    torch.set_float32_matmul_precision("medium")
    # Enable TF32 on matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    test_loaders = get_test_loaders(cfg.experiment.test_data_paths)

    module = SimpleModule.load_from_checkpoint(cfg.model_ckpt_path)

    trainer = L.Trainer(logger=False)

    for test_loader in test_loaders:
        trainer.test(module, dataloaders=test_loader)


def get_test_loaders(paths):
    test_datasets = [AllenCahnInMemoryDataset(data_path) for data_path in paths]
    test_dataloaders = [DataLoader(dataset, batch_size=128) for dataset in test_datasets]
    return test_dataloaders


if __name__ == "__main__":
    main()
