import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.mlp import MLP
from mondrian_lib.models.fdm.dd_op import DDOp

class DDFNO(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes):
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
        
        self.sc1 = DDOp(SpectralConv(self.hidden_channels, self.hidden_channels, n_modes), 1, 1, 0.1)
        self.sc2 = DDOp(SpectralConv(self.hidden_channels, self.hidden_channels, n_modes), 1, 1, 0.1)
        self.sc3 = DDOp(SpectralConv(self.hidden_channels, self.hidden_channels, n_modes), 1, 1, 0.1)
    
    def forward(self, x, xlim, ylim):
        x = self.lifting(x)
        x = self.sc1(x, xlim, ylim)
        x = self.sc2(x, xlim, ylim)
        x = self.sc3(x, xlim, ylim)
        x = self.projection(x)
        return x


if __name__ == '__main__':
    dd = DDFNO(2, 2, (8, 8))
    i = torch.ones((2, 2, 16, 16))
    dd(i, 2, 2)
