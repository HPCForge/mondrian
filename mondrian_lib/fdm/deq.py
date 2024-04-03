import torch
import torch.nn as nn
import torchdeq
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import ctypes

mod = True
debug = False

"""
TODO:

    1. Use Arthur's FDM model with the torchdeq library
    2. Implement my own model?
    3. Implement PoissonDEQ for 2D
    4. Implement PoissonDEQ for 3D

"""

"""
    1. Use Arthur's FDM model with the torchdeq library 
"""

from mondrian_lib.fdm.deq_dd_fno import DDFNO
from neuralop.layers.mlp import MLP
from torchdeq.core import get_deq, register_deq

# Implement the DDFNO as a DEQ model
class DEQ_DDFNO(nn.Module):
    """
    CTOR:
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 f_max_iter,
                 n_modes):
        
        super().__init__()
        self.n_dim = len(n_modes)


        self.f_max_iter = f_max_iter
        self.hidden_channels = 32
        self.lifting_channels = 128
        self.projection_channels = 128

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ddfno = DDFNO(in_channels, out_channels, n_modes)
        register_deq("DEQ_DDFNO", DEQ_DDFNO)
        self.deq = get_deq(core="sliced")

        self.lifting = MLP(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
        )

        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
        )

    """
    Forward():
    """
    def forward(self, x, xlim, ylim):
        """
        1. DDFNO is the DEQ function
        """

        # Dimensions
        if debug:
            print("DEQ Forward():")

            print("\tx: ", x.shape)
            print("\txlim: ", xlim.shape)
            print("\tylim: ", ylim.shape)

            print("\n\tx = self.lifting(x)")
        if mod: x = self.lifting(x)
        if debug: print("\tx: ", x.shape)

        def f(_x):
            return self.ddfno(_x, xlim, ylim)
        
        if debug: print("\n\tx = self.deq(x, f)")
        x, info = self.deq(f, x)
        x = x[0]
        if debug: 
            print("\tx: ", x.shape)
            print("\n\tx = self.projection(x)")
        if mod: x = self.projection(x)
        if debug: print("\tx.shape: ", x.shape)
        return x

