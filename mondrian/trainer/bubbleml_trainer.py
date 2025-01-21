import torch
import torch.nn.functional as F

import lightning as L

from mondrian.metrics import Metrics
from mondrian.lr_schedule import WarmupCosineAnnealingLR
from mondrian.dataset.bubbleml.constants import (
    unnormalize_velx,
    unnormalize_vely,
    unnormalize_temperature
)

def unnormalize_var_metrics(pred, target):
    assert target.size(1) % 4 == 0 
    s = target.size(1) // 4
    velx_pred = unnormalize_velx(pred[:, :s])
    velx_target = unnormalize_velx(target[:, :s])
    velx_loss = F.mse_loss(velx_pred, velx_target)
    
    vely_pred = unnormalize_vely(pred[:, :s])
    vely_target = unnormalize_vely(target[:, :s])
    vely_loss = F.mse_loss(vely_pred, vely_target)
    
    temp_pred = unnormalize_temperature(pred[:, 2*s : 3*s].detach().cpu().numpy())
    temp_target = unnormalize_temperature(target[:, 2*s : 3*s].detach().cpu().numpy())
    temp_loss = F.mse_loss(torch.from_numpy(temp_pred), torch.from_numpy(temp_target))
    
    mask_pred = pred[:, 3*s:]
    mask_target = target[:, 3*s:]
    mask_loss = F.mse_loss(mask_pred, mask_target)
    
    return velx_loss, vely_loss, temp_loss, mask_loss

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

    def _log_vars(self, pred, target, stage):
        velx_loss, vely_loss, temp_loss, mask_loss = unnormalize_var_metrics(pred, target)
        self.log(f'{stage}/MSE-velx', velx_loss)
        self.log(f'{stage}/MSE-vely', vely_loss)
        self.log(f'{stage}/MSE-temp', temp_loss)
        self.log(f'{stage}/MSE-mask', mask_loss)

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