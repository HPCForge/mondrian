import torch
import torch.nn.functional as F

import lightning as L

from mondrian.metrics import Metrics
from mondrian.lr_schedule import WarmupCosineAnnealingLR


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
        # loss = self.metrics.log(pred, y, "Train")
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
        return loss
