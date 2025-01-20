import torch
import torch.nn.functional as F

import lightning as L

from mondrian.metrics import Metrics
from mondrian.lr_schedule import WarmupCosineAnnealingLR

class BubbleMLModule(L.LightningModule):
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

    def forward(self, x, nuc):
        if self.domain_size is not None:
            return self.model(x, nuc, self.domain_size[0], self.domain_size[1])
        else:
            return self.model(x, nuc)

    def _xvel_mse_loss(self, pred, target, stage):
        s = target.size(0) // 4
        loss = F.mse_loss(pred[0:s], target[0:s])
        self.log(f'{stage}/MSE-xvel', loss)

    def _yvel_mse_loss(self, pred, target, stage):
        s = target.size(0) // 4
        loss = F.mse_loss(pred[s:2*s], target[s:2*s])
        self.log(f'{stage}/MSE-yvel', loss)
    
    def _temp_mse_loss(self, pred, target, stage):
        s = target.size(0) // 4
        loss = F.mse_loss(pred[2*s:3*s], target[2*s:3*s])
        self.log(f'{stage}/MSE-temp', loss)
    
    def _mask_mse_loss(self, pred, target, stage):
        s = target.size(0) // 4
        loss = F.mse_loss(pred[3*s:], target[3*s:])
        self.log(f'{stage}/MSE-bubbleml-mask', loss)

    def _log_vars(self, pred, target, stage):
        self._xvel_mse_loss(pred, target, stage)
        self._yvel_mse_loss(pred, target, stage)
        self._temp_mse_loss(pred, target, stage)
        self._dfun_mse_loss(pred, target, stage)

    def training_step(self, batch, batch_idx):
        x, nuc, y = batch
        pred = self.forward(x, nuc)
        loss = self.metrics.log(pred, y, "Train")
        self._log_vars(pred, y, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, nuc, y = batch
        pred = self.forward(x, nuc)
        loss = self.metrics.log(pred, y, "Val")
        self._log_vars(pred, y, "Val")
        return loss

    def test_step(self, batch, batch_idx):
        x, nuc, y = batch
        pred = self.forward(x, nuc)
        loss = self.metrics.log(pred, y, "Test")
        self._log_vars(pred, y, "Test")
        return loss