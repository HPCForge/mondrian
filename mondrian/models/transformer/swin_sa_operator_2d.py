from typing import Union, Tuple

import einops
import torch
from torch import nn

from mondrian.grid.decompose import win_decompose2d, win_recompose2d
from mondrian.grid.spectral_conv import SimpleSpectralConv2d
from mondrian.attention.swin_func_self_attention import SwinFuncSelfAttention
from mondrian.grid.pointwise import PointwiseMLP2d
from mondrian.grid.seq_op import seq_op
from mondrian.grid.utility import cell_centered_grid

class SequenceInstanceNorm2d(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.norm = nn.InstanceNorm2d(embed_dim)
    
  def forward(self, v):
    return seq_op(self.norm, v)
  
class Encoder(nn.Module):
  def __init__(self, embed_dim, num_heads, head_split, score_method, use_bias, shift_size, n_sub, window_size):
    super().__init__()
    self.shift_size = shift_size
    self.sa = SwinFuncSelfAttention(
      embed_dim, num_heads, head_split, use_bias, shift_size, n_sub, window_size, score_method
    )
    self.mlp = PointwiseMLP2d(embed_dim, embed_dim, embed_dim)
    self.norm1 = SequenceInstanceNorm2d(embed_dim)
    self.norm2 = SequenceInstanceNorm2d(embed_dim)
  
  def forward(self, v):
        v = self.sa(self.norm1(v)) + v
        v = seq_op(self.mlp, self.norm2(v)) + v
        return v

class SwinSAOperator2d(nn.Module):
  r""" 
  A shifted windowed self-attention operator for 2D data, modified from ViTSelfAttentionOperator by adding windowize self-attention.
  Removed n_sub_x, n_sub_y from parameter as self-attention is calculated in window level.
  Parameters:
    in_channels: The expected number of channels input to the model.
    out_channels: The number of channels output by the model. 
    embed_dim: The number of channels used in the attention operators. 
    num_heads: The number of heads used in multihead attention.
    head_split: way to split heads for multihead attention. ['spatial', 'channel'] 
    num_layers: The number of Encoder blocks.
    window_size: The number of the subdomain in each coordinate of the window (assuming square window).
    shift_size: The number of the subdomain to shift in each coordinate of the window.
    n_sub: The number of subdomains in each coordinate of the global domain.
    sub_domain_size: The physical subdomain size. This is independent of
                    the input discretization. It should correspond to some
                    "physical" dimension, relative to the global domain size.
  """
  def __init__(self,
               in_channels: int,
               out_channels: int,
               embed_dim: int,
               num_heads: int,
               head_split: str,
               score_method: str,
               num_layers: int,
               window_size: int,
               shift_size: int,
               n_sub: int,
               subdomain_size: Union[int, Tuple[int, int]]):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    if isinstance(subdomain_size, int):
      subdomain_size = (subdomain_size, subdomain_size)
    assert isinstance(subdomain_size[0], int) 
    assert isinstance(subdomain_size[1], int)

    self.n_sub = n_sub
    self.shift_size = shift_size
    self.window_size = window_size
    self.subdomain_size = subdomain_size
    self.sub_size_y = self.subdomain_size[0]
    self.sub_size_x = self.subdomain_size[1]
    
    self.encoder = nn.ModuleList([
      Encoder(embed_dim, 
              num_heads, 
              head_split,
              score_method,
              True, 
              shift_size=0 if (i%2==0) else self.shift_size , 
              n_sub=self.n_sub, 
              window_size=self.window_size) 
      for i in range(num_layers)
    ])
    
    self.input_project = PointwiseMLP2d(
            in_channels + 2, embed_dim, hidden_channels=128
    )
    self.output_project = PointwiseMLP2d(embed_dim, out_channels, hidden_channels=128)
    
  def flatten(self, v):
    r"""
    flatten batch and sequence dimensions into one dimension
    Args:
      v: [batch x seq-len x ...]
    Returns:
      flat: [(batch x seq_len) x ...]]
    """
    return torch.flatten(v, start_dim=0, end_dim=1)
    
  def forward(self, v: torch.Tensor, domain_size_y: int, domain_size_x: int):
    assert v.size(1) == self.in_channels
    assert isinstance(domain_size_y, int)
    assert isinstance(domain_size_x, int)
    assert domain_size_y % self.sub_size_y == 0
    assert domain_size_x % self.sub_size_x == 0
    n_sub_y = domain_size_y // self.sub_size_y
    n_sub_x = domain_size_x // self.sub_size_x

    # concatenate point-wise positions
    height = v.size(-2)
    width = v.size(-1)
    g = cell_centered_grid(
        (height, width), (domain_size_y, domain_size_x), device=v.device
    )
    g = einops.repeat(g, "... -> b ...", b=v.size(0))
    v = torch.cat((g, v), dim=1)


    v = self.input_project(v)
    d = win_decompose2d(v, n_sub_x, n_sub_y, self.window_size)

    for encoder in self.encoder:
      d = encoder(d)
      
    u = win_recompose2d(d, n_sub_x, n_sub_y, self.window_size)
    u = self.output_project(u)
      
    return u
  