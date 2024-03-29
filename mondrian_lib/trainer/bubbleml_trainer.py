import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from mondrian_lib.data.bubbleml_dataset import BubbleMLDataset
from mondrian_lib.data.data_loaders import get_data_loaders 
from mondrian_lib.fdm.dd_fno import DDFNO
import numpy as np
from neuralop.losses import LpLoss
import matplotlib
import matplotlib.pyplot as plt

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
