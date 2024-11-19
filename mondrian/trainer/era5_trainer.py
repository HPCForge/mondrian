from typing import Tuple, List

import torch
import torch.nn.functional as F
import lightning as L

from mondrian.lr_schedule import WarmupCosineAnnealingLR
from climate_learn.metrics import (
  MSE,
  RMSE,
  ACC,
  Pearson
)

class ERA5Module(L.LightningModule):
    def __init__(self,
                 model,
                 total_iters,
                 train_denormalize,
                 val_denormalize,
                 test_denormalize,
                 lr=0.001,
                 weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.total_iters = total_iters
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.train_denormalize = train_denormalize
        self.val_denormalize = val_denormalize
        self.test_denormalize = test_denormalize
        
        self.loss = MSE(aggregate_only=True)
        self.metrics_dict = {
          'MSE': MSE(),
          'RMSE': RMSE(),
          'Pearson': Pearson()
        }
    
    def metrics(self, pred, label, stage, out_vars):
      r"""
      Compute channel-wise metrics. These correspond to errors
      for the different out_vars.
      All metrics are computed on denormalized data.
      """
      pred = self.train_denormalize(pred)
      label = self.train_denormalize(label)
      for metric_name, metric in self.metrics_dict.items():
        err = metric(pred, label)
        # Metrics output a tensor with err[-1] being the total aggregate error.
        # The first [0:-1] errors are the channel-wise error corresponding to an out_var
        for out_var, idx in zip(out_vars, range(err.size(0) - 1)):
          self.log(f'{stage}/{out_var}/{metric_name}', err[idx])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = WarmupCosineAnnealingLR(
                optimizer,
                warmup_iters=1000,
                total_iters=self.total_iters,
                eta_min=1e-8)
        scheduler_config = {'scheduler': scheduler,
                            'interval': 'epoch'}
        return [optimizer], [scheduler_config]

    def forward(self, x):
        return self.model(x, 4, 8)

    def training_step(self, 
                      batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
                      batch_idx: int):
        x, y, in_vars, out_vars = batch
        x = x.flatten(start_dim=1, end_dim=2)
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log('Train/MSELoss', loss, prog_bar=True)
        
        self.metrics(pred, y, 'Train', out_vars)
        return loss

    def validation_step(self, 
                        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
                        batch_idx: int):
        x, y, in_vars, out_vars = batch
        x = x.flatten(start_dim=1, end_dim=2)
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log('Val/MSELoss', loss, prog_bar=True)
        self.metrics(pred, y, 'Val', out_vars) 
        return loss

    def test_step(self, 
                  batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
                  batch_idx: int):
        x, y, in_vars, out_vars = batch
        x = x.flatten(start_dim=1, end_dim=2)
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log('Test/MSELoss', loss, prog_bar=True)
        self.metrics(pred, y, 'Test', out_vars)
        return loss