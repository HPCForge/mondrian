import einops
import torch
import torch.nn as nn
from mondrian_lib.fdm.models.ddno.self_attention import SelfAttention
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models.fno import FNO

class Encoder(nn.Module):
    def __init__(self,
                 sequence_length,
                 hidden_channels,
                 heads):
        super().__init__()
        
        self.sa = SelfAttention(sequence_length,
                                hidden_channels,
                                hidden_channels,
                                heads)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.n_modes = (16, 16)
        self.op = FNO(self.n_modes,
                      hidden_channels=hidden_channels,
                      in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      n_layers=2)

    def apply_op(self, v):
        h = torch.flatten(v, start_dim=0, end_dim=1)
        h = self.op(h)
        h = torch.unflatten(h, dim=0, sizes=(v.size(0), v.size(1)))
        return h

    def apply_norm(self, v, norm):
        # Only apply layer norm along the channel dim
        h = einops.rearrange(v, 'b s d h w -> b s h w d')
        h = norm(h)
        h = einops.rearrange(h, 'b s h w d -> b s d h w')
        return h

    def forward(self, v):
        r"""
        Compute transformer encoder of v 
        Args:
            v: [batch, sequence-length, hidden_channels, H, W]
        Returns:
            u: [batch, sequence-length, hidden_channels, H, W]
        """
        assert v.dim() == 5

        h = self.sa(self.apply_norm(v, self.norm1))
        u = self.apply_op(self.apply_norm(h, self.norm2)) + h

        return u 
