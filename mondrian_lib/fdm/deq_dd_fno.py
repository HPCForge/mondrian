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
        in_channels,
        out_channels,
        n_modes
    ):
        super().__init__()
        self.n_dim = len(n_modes)

        self.in_channels = in_channels * 2
        self.out_channels = out_channels
        self.hidden_channels = 64
        self.lifting_channels = 128
        self.projection_channels = 128

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.ln = [MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim).to(self.device) for _ in range(3)]

        DDOp = DDOpAdditive

        self.lifting = MLP(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=1,
            n_dim=self.n_dim,
        )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=1,
            n_dim=self.n_dim,
        )

        def dd_op(use_coarse_op=True):
            return DDOp(SpectralConv(self.hidden_channels,
                                     self.hidden_channels, 
                                     n_modes),
                        self.hidden_channels, 1, 1, 0.2, 0.2, use_coarse_op=use_coarse_op, use_padding=True)
        
        self.scn = [dd_op().to(self.device) for _ in range(3)]

    def forward(self, x, injection=None, xlim=None, ylim=None):
        x = torch.cat([x, injection], dim=1)
        x = self.lifting(x)
        for sc, l in zip(self.scn, self.ln):
            x = F.gelu(sc(x, xlim[0].item(), ylim[0].item())) + l(x)
        x = self.projection(x)
        return x
