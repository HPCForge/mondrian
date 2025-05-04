from functools import partial

import torch
import torch.nn.functional as F
from neuralop.losses.data_losses import LpLoss
import lightning as L

from mondrian.metrics import Metrics
from mondrian.lr_schedule import WarmupCosineAnnealingLR
from mondrian.dataset.bubbleml.constants import (
    unnormalize_velx,
    unnormalize_vely,
    unnormalize_temperature
)

def relative_mse(pred, target):
    loss = LpLoss(d=2, reduce_dims=(0, 1), reductions='mean')
    loss = loss.rel(pred, target)
    return loss

def unnormalize_var_metrics(pred, target):
    assert target.size(1) % 4 == 0 
    s = target.size(1) // 4
    velx_pred = unnormalize_velx(pred[:, :s])
    velx_target = unnormalize_velx(target[:, :s])
    velx_loss = F.mse_loss(velx_pred, velx_target)
    
    vely_pred = unnormalize_vely(pred[:, s:2*s])
    vely_target = unnormalize_vely(target[:, s:2*s])
    vely_loss = F.mse_loss(vely_pred, vely_target)
    
    temp_pred = unnormalize_temperature(pred[:, 2*s : 3*s].detach().cpu().numpy())
    temp_target = unnormalize_temperature(target[:, 2*s : 3*s].detach().cpu().numpy())
    temp_loss = F.mse_loss(torch.from_numpy(temp_pred), torch.from_numpy(temp_target))
    
    mask_pred = pred[:, 3*s:]
    mask_target = target[:, 3*s:]
    mask_loss = F.mse_loss(mask_pred, mask_target)
    
    return velx_loss, vely_loss, temp_loss, mask_loss

def var_losses(pred, target):
    assert target.size(1) % 4 == 0 
    s = target.size(1) // 4
    velx_loss = relative_mse(pred[:, :s], target[:, :s])
    vely_loss = relative_mse(pred[:, s:2*s], target[:, s:2*s])
    temp_loss = relative_mse(pred[:, 2*s:3*s], target[:, 2*s:3*s])
    mask_loss = relative_mse(pred[:, 3*s:], target[:, 3*s:])
    return (velx_loss + vely_loss + temp_loss + mask_loss)

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
        self.log_func = partial(self.log)
        self.metrics = Metrics(self.log_func)

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
        self.log_func(f'{stage}/MSE-velx', velx_loss)
        self.log_func(f'{stage}/MSE-vely', vely_loss)
        self.log_func(f'{stage}/MSE-temp', temp_loss)
        self.log_func(f'{stage}/MSE-mask', mask_loss)

    def training_step(self, batch, batch_idx):
        x, nuc, y = batch
        pred = self.forward(x, nuc)
        self.metrics.log(pred, y, "Train")
        self._log_vars(pred, y, "Train")
        loss = var_losses(pred, y)
        self.log_func(f'Train/rel-loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, nuc, y = batch
        pred = self.forward(x, nuc)
        self.metrics.log(pred, y, "Val")
        self._log_vars(pred, y, "Val")
        loss = var_losses(pred, y)
        self.log_func(f'Val/rel-loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, nuc, y = batch
        pred = self.forward(x, nuc)
        self.metrics.log(pred, y, "Test")
        self._log_vars(pred, y, "Test")
        loss = var_losses(pred, y)
        self.log_func(f'Test/rel-loss', loss)
        return loss