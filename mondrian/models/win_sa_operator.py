from typing import Union, Tuple

import torch
from torch import nn

from mondrian.grid.decompose import win_decompose2d, win_recompose2d
from mondrian.grid.spectral_conv import SimpleSpectralConv2d
from mondrian.grid.func_self_attention import WinFuncSelfAttention
from mondrian.grid.pointwise import PointwiseMLP2d
from mondrian.grid.seq_op import seq_op
from mondrian.grid.pos_embedding import FuncPosEmbedding2d

from neuralop.layers.padding import DomainPadding

class SequenceInstanceNorm2d(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.norm = nn.InstanceNorm2d(embed_dim)
    
  def forward(self, v):
    return seq_op(self.norm, v)
  
class Encoder(nn.Module):
  def __init__(self, 
               embed_dim, 
               num_heads,
               use_bias,
               shift_size,
               n_sub,
               window_size):
    super().__init__()
    self.shift_size = shift_size
    self.sa = WinFuncSelfAttention(embed_dim, num_heads, use_bias, shift_size,  n_sub, window_size)
    modes = 8
    self.spectral_conv1 = SimpleSpectralConv2d(embed_dim, embed_dim, modes)
    self.spectral_conv2 = SimpleSpectralConv2d(embed_dim, embed_dim, modes)

    self.norm1 = SequenceInstanceNorm2d(embed_dim)
    self.norm2 = SequenceInstanceNorm2d(embed_dim)
  
  def _spectral(self, v):
    v = nn.functional.gelu(self.spectral_conv1(v)) + v
    v = self.norm2(self.spectral_conv2(v) + v)
    return v
  
  def forward(self, v):
    v = self.norm1(self.sa(v) + v)
    v = nn.functional.gelu(self.spectral_conv1(v) + v)
    v = self._spectral(self.norm2(v)) + v
    return v

class SwinSAOperator2d(nn.Module):
  r""" 
  A windowed self-attention operator for 2D data, modified from ViTSelfAttentionOperator by adding windowize self-attention.
  Removed n_sub_x, n_sub_y from parameter as self-attention is calculated in window level.
  Parameters:
  in_channels: The expected number of channels input to the model.
      out_channels: The number of channels output by the model. 
      embed_dim: The number of channels used in the attention operators. 
      num_heads: The number of heads used in multihead attention. 
      num_layers: The number of Encoder blocks.
      window_size: The number of the subdomain in each coordinate of the window.
      sub_domain_size: The physical subdomain size. This is independent of
                       the input discretization. It should correspond to some
                       "physical" dimension, relative to the global domain size.
  """
  def __init__(self,
               in_channels,
               out_channels,
               embed_dim,
               num_heads,
               num_layers,
               window_size,
               shift_size,
               n_sub,
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
    
    self.embed = nn.ModuleList([
      Encoder(embed_dim, 
              num_heads, 
              True, 
              shift_size=0 if (i%2==0) else self.shift_size , 
              n_sub=self.n_sub, 
              window_size=self.window_size) 
      for i in range(num_layers)
    ])
    
    self.input_project = PointwiseMLP2d(in_channels, embed_dim, hidden_channels=128)
    self.output_project = PointwiseMLP2d(embed_dim, out_channels, hidden_channels=128)
    
    # TODO: Maybe make this optional...
    self.pos_embedding = FuncPosEmbedding2d(max_seq_len=64, channels=embed_dim)
    self.padding = DomainPadding(0.25)
    
  def flatten(self, v):
    r"""
    flatten batch and sequence dimensions into one dimension
    Args:
      v: [batch x seq-len x ...]
    Returns:
      flat: [(batch x seq_len) x ...]]
    """
    return torch.flatten(v, start_dim=0, end_dim=1)
  
  def unflatten(self, v, batch_size, seq_len):
    return torch.unflatten(v, dim=0, sizes=(batch_size, seq_len))
    
  def forward(self, v: torch.Tensor, domain_size_y: int, domain_size_x: int):
    assert v.size(1) == self.in_channels
    assert isinstance(domain_size_y, int)
    assert isinstance(domain_size_x, int)
    assert domain_size_y % self.sub_size_y == 0
    assert domain_size_x % self.sub_size_x == 0
    n_sub_y = domain_size_y // self.sub_size_y
    n_sub_x = domain_size_x // self.sub_size_x

    v = self.input_project(v)
    d = win_decompose2d(v, n_sub_x, n_sub_y, self.window_size)
    p = self.pos_embedding(d)
    d = d + p
    d = seq_op(self.padding.pad, d)

    for embed in self.embed:
      d = embed(d)
      
    d = seq_op(self.padding.unpad, d)
    u = win_recompose2d(d, n_sub_x, n_sub_y, self.window_size)
    u = self.output_project(u)
      
    return u