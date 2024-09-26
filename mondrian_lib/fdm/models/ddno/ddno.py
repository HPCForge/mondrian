import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.mlp import MLP
from neuralop.layers.skip_connections import skip_connection

from mondrian_lib.fdm.dd_op.dd_op_additive import DDOpAdditive
from mondrian_lib.fdm.dd_op.dd_op_alternating import DDOpAlternating
from mondrian_lib.fdm.kernel.low_rank import LowRankKernel
from mondrian_lib.fdm.kernel.nonlinear import NonLinearKernel
from mondrian_lib.fdm.kernel.linear import LinearKernel
from mondrian_lib.fdm.kernel.nystrom_nonlinear import NystromNonLinearKernel
from mondrian_lib.fdm.kernel.nystrom_linear import NystromLinearKernel
from mondrian_lib.fdm.build_op_from_cfg import build_op_from_cfg

class DDNO(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        layers: int,
        domain_size_x: float,
        domain_size_y: float,
        op_cfg,
        lifting_channels=256,
        projection_channels=256,
    ):
        super().__init__()
        self.n_dim = 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels

        self.layers = layers

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

        self.skip = nn.ModuleList([
                skip_connection(self.hidden_channels,
                                self.hidden_channels,
                                skip_type='linear',
                                n_dim=self.n_dim)
                for _ in range(self.layers)
            ])

        # GroupNorm paper uses groups of size 8 or 16.
        num_groups = self.hidden_channels // 8
        self.norm = nn.ModuleList([
                nn.GroupNorm(num_groups=num_groups,
                             num_channels=self.hidden_channels)
                for _ in range(self.layers)
            ])

        self.ddop = nn.ModuleList([
                # initialize DDOp using subdomain_kernel_kwargs 
                build_op_from_cfg(op_cfg,
                                  in_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  hidden_channels=hidden_channels,
                                  domain_size_x=domain_size_x,
                                  domain_size_y=domain_size_y,
                                  use_coarse_op=op_cfg.use_coarse_op)
                for _ in range(self.layers)
            ])

    def forward(self, x):
        x = self.lifting(x)

        for i in range(self.layers):
            h = self.ddop[i](x)
            #h = self.norm[i](h)
            h = F.gelu(h)
            x = h + self.skip[i](x)

        x = self.projection(x)
        return x
