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

        self.ln = [MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim) for _ in range(5)]
        
        DDOp = DDOpAdditive

        def dd_op(use_coarse_op=True):
            return DDOp(SpectralConv(self.hidden_channels,
                                     self.hidden_channels, 
                                     n_modes),
                        self.hidden_channels, 1, 1, 0.2, 0.2, use_coarse_op=use_coarse_op, use_padding=True)
        
        self.scn = [dd_op() for _ in range(4)]

    def forward(self, x, xlim, ylim):
        for sc, l in zip(self.scn, self.ln):
            x = F.gelu(sc(x, xlim, ylim)) + l(x)
        return x
