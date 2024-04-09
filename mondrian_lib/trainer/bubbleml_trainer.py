import torch
import torch.nn.functional as F

import lightning as L

class BubbleMLModule(L.LightningModule):
    def __init__(self, model, total_iters, loss_func=F.mse_loss):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.total_iters = total_iters

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=self.total_iters)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('Train/Loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('Val/Loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('Test/Loss', loss)
        return loss
