import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.mlp import MLP
from mondrian_lib.fdm.dd_op.dd_op_additive import DDOpAdditive
from mondrian_lib.fdm.dd_op.dd_op_alternating import DDOpAlternating
from mondrian_lib.fdm.kernel.low_rank import LowRankKernel
from mondrian_lib.fdm.kernel.nonlinear import NonLinearKernel
from mondrian_lib.fdm.kernel.linear import LinearKernel
from mondrian_lib.fdm.kernel.nystrom_nonlinear import NystromNonLinearKernel

class DDFNO(nn.Module):
    def __init__(
        self,
        n_modes
    ):
        super().__init__()
        self.n_dim = len(n_modes)

        self.hidden_channels = 32

        self.l1 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        self.l2 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        self.l3 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        self.l4 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        self.l5 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        
        DDOp = DDOpAdditive

        def dd_op(use_coarse_op=True):
            return DDOp(SpectralConv(self.hidden_channels,
                                     self.hidden_channels, 
                                     n_modes),
                        self.hidden_channels, 1, 1, 0.2, 0.2, use_coarse_op=use_coarse_op, use_padding=True)

        self.sc1 = dd_op() 
        self.sc2 = dd_op()
        self.sc3 = dd_op()
        self.sc4 = dd_op()

    def forward(self, x, xlim, ylim):
        x = F.gelu(self.sc1(x, xlim, ylim)) + self.l1(x)
        x = F.gelu(self.sc2(x, xlim, ylim)) + self.l2(x)
        x = F.gelu(self.sc3(x, xlim, ylim)) + self.l3(x)
        x = F.gelu(self.sc4(x, xlim, ylim)) + self.l4(x)
        return x
