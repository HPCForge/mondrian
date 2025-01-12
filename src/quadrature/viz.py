from omegaconf import OmegaConf
import hydra
import pathlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.simple_trainer import SimpleModule
from mondrian.dataset.allen_cahn import AllenCahnInMemoryDataset

from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # enable using tensor cores
    torch.set_float32_matmul_precision("high")
    # Enable TF32 on matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    module = SimpleModule.load_from_checkpoint(cfg.model_ckpt_path)
    module = module.cpu()
    
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)

    test_datasets = [
        AllenCahnInMemoryDataset(data_path)
        for data_path in cfg.experiment.test_data_paths
    ]
    sizes = [32, 64, 128]

    with torch.no_grad():
        for size, dataset in zip(sizes, test_datasets):
            for i in range(3):
                batch = dataset[i]
                input, label = batch
                pred = module(input.unsqueeze(0))
                fig, axarr = plt.subplots(1, 3, figsize=(7, 2))
                axarr[0].imshow(input[0], cmap="plasma")
                axarr[1].imshow(pred.squeeze().detach(), cmap="plasma")
                axarr[2].imshow(label.squeeze(), cmap="plasma")    
                plt.tight_layout()
                plt.savefig(f"ac_{size}.png")


if __name__ == "__main__":
    main()
