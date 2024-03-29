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

        self.hidden_channels = 32
        self.lifting_channels = 128
        self.projection_channels = 128

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

        self.l1 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        self.l2 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        self.l3 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        self.l4 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        self.l5 = MLP(self.hidden_channels, self.hidden_channels, self.hidden_channels, n_layers=1, n_dim=self.n_dim)
        
        DDOp = DDOpAdditive

        def dd_op(use_coarse_op=True):
            #return DDOp(NystromNonLinearKernel(self.hidden_channels, self.hidden_channels, self.hidden_channels, 3),
            #            self.hidden_channels, 1, 1, 0.2, 0.2, use_coarse_op=use_coarse_op, use_padding=False)
            #return DDOp(LinearKernel(self.hidden_channels, self.hidden_channels, self.hidden_channels, 4),
            #            self.hidden_channels, 1, 1, 0.2, 0.2, use_coarse_op=use_coarse_op, use_padding=False)
            #return DDOp(NonLinearKernel(self.hidden_channels, self.hidden_channels, self.hidden_channels, 4),
            #            self.hidden_channels, 1, 1, 0.2, 0.2, use_coarse_op=use_coarse_op, use_padding=False)
            #return DDOp(LowRankKernel(self.hidden_channels, self.hidden_channels, self.hidden_channels, 8),
            #            self.hidden_channels, 1, 1, 0.2, 0.2, use_coarse_op=use_coarse_op, use_padding=False)
            return DDOp(SpectralConv(self.hidden_channels,
                                     self.hidden_channels, 
                                     n_modes),
                        self.hidden_channels, 1, 1, 0.2, 0.2, use_coarse_op=use_coarse_op, use_padding=True)

        self.sc1 = dd_op() 
        self.sc2 = dd_op()
        self.sc3 = dd_op()
        self.sc4 = dd_op()
        #self.sc5 = dd_op(use_coarse_op=False)

        """
        self.n1 = nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels)
        self.n2 = nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels)
        self.n3 = nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels)
        self.n4 = nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels)
        self.n5 = nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels)
        """
        #self.n1 = nn.InstanceNorm2d(num_features=self.hidden_channels)
        #self.n2 = nn.InstanceNorm2d(num_features=self.hidden_channels)
        #self.n3 = nn.InstanceNorm2d(num_features=self.hidden_channels)
        #self.n4 = nn.InstanceNorm2d(num_features=self.hidden_channels)
        #self.n5 = nn.InstanceNorm2d(num_features=self.hidden_channels)

    def forward(self, x, xlim, ylim):
        x = self.lifting(x)
        x = F.gelu(self.sc1(x, xlim, ylim)) + self.l1(x)
        x = F.gelu(self.sc2(x, xlim, ylim)) + self.l2(x)
        x = F.gelu(self.sc3(x, xlim, ylim)) + self.l3(x)
        x = F.gelu(self.sc4(x, xlim, ylim)) + self.l4(x)
        #x = F.gelu(self.sc5(x, xlim, ylim)) + self.l5(x)
        x = self.projection(x)
        return x
