import torch
from torch import nn
from neuralop.layers.spectral_convolution import SpectralConv

class SimpleSpectralConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_modes,
                 version='fno'):
        super().__init__()
        # TODO: reenable FFNO or similar
        Spectral = SpectralConv
        if isinstance(n_modes, int):
            n_modes = (n_modes, n_modes)
        
        self.spectral = Spectral(in_channels, out_channels, n_modes)
        
    def _unflatten(self, f, batch_size, seq_len):
        return torch.unflatten(f, 0, (batch_size, seq_len))
    
    def _flatten(self, f):
        return torch.flatten(f, start_dim=0, end_dim=1)

    def forward(self, x):
        # NOTE: SpectalConv doesn't like [batch x seq x ...] layout. 
        # Need to flatten into [(batch x seq) x ...]
        batch_size, seq_len = x.size(0), x.size(1)
        x = self._flatten(x)
        x = self.spectral(x)
        return self._unflatten(x, batch_size, seq_len)