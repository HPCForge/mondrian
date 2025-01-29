import torch
import torch.nn.functional as F

import lightning as L

from mondrian.metrics import Metrics
from mondrian.lr_schedule import WarmupCosineAnnealingLR

def interface_mse(pred, target):
    r"""
    This metric is specific to allen-cahn. Allen-cahn has metastable states
    of +1, -1. And there is a thing interface between the two states. The size of the interface
    is determined by the parameter $\gamma$ that is passed to the model. 
    Since most of the output is a few blobs of +1 or -1 a global MAE can drown out errors along the interface.
    """
    interface_mask = (target < 0.98) & (target > -0.98)

    target[~interface_mask] = 0
    pred[~interface_mask] = 0

    return (((target - pred) ** 2).sum(dim=(-3, -2, -1)) / interface_mask.sum(dim=(-3, -2, -1))).mean()


class SimpleModule(L.LightningModule):
    def __init__(
        self,
        model,
        total_iters,
        domain_size,
        lr=0.001,
        weight_decay=1e-4,
        warmup_iters=1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.total_iters = total_iters
        self.domain_size = domain_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.metrics = Metrics(self.log)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_iters=self.warmup_iters,
            total_iters=self.total_iters,
            eta_min=1e-8,
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def forward(self, x):
        if self.domain_size is not None:
            return self.model(x, self.domain_size[0], self.domain_size[1])
        else:
            return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = F.mse_loss(pred, y)
        loss = self.metrics.log(pred, y, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.metrics.log(pred, y, "Val")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.metrics.log(pred, y, "Test")
        self.log('Test/InterfaceMSE', interface_mse(pred, y))
        return loss
