from typing import Union, Tuple

import torch
from torch import nn

from mondrian.grid.decompose import decompose2d, recompose2d
from mondrian.grid.spectral_conv import SimpleSpectralConv2d
from mondrian.grid.vit_self_attention_operator import ViTSelfAttentionOperator
from mondrian.grid.pointwise_linear import PointwiseLinear
from mondrian.grid.seq_op import seq_op

from neuralop.layers.padding import DomainPadding

class SequenceInstanceNorm2d(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.norm = nn.InstanceNorm2d(embed_dim)
    
  def forward(self, v):
    r"""
    Flattens the batch and sequence dimension of the input
    `v`, so that it can be passed into a standard instance norm.
    """
    return seq_op(self.norm, v)

class Encoder(nn.Module):
  def __init__(self, 
               embed_dim, 
               num_heads):
    super().__init__()
    self.sa = ViTSelfAttentionOperator(embed_dim, num_heads)
    modes = 16
    self.spectral_conv1 = SimpleSpectralConv2d(embed_dim, embed_dim, modes)
    self.spectral_conv2 = SimpleSpectralConv2d(embed_dim, embed_dim, modes)

    self.norm1 = SequenceInstanceNorm2d(embed_dim)
    self.norm2 = SequenceInstanceNorm2d(embed_dim)
    
  def forward(self, v, n_sub_x, n_sub_y):
    v = self.norm1(self.sa(v, n_sub_x, n_sub_y) + v)
    v = nn.functional.gelu(self.spectral_conv1(v) + v)
    v = self.norm2(self.spectral_conv2(v) + v)
    return v

class ViTOperator2d(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               embed_dim,
               num_heads,
               num_layers,
               subdomain_size: Union[int, Tuple[int, int]]):
    super().__init__()
    
    if isinstance(subdomain_size, int):
      subdomain_size = (subdomain_size, subdomain_size)
    assert isinstance(subdomain_size[0], int) 
    assert isinstance(subdomain_size[1], int)

    self.subdomain_size = subdomain_size
    self.sub_size_y = self.subdomain_size[0]
    self.sub_size_x = self.subdomain_size[1]
    
    self.embed = nn.ModuleList([
      Encoder(embed_dim, num_heads) for _ in range(num_layers)
    ])
    
    self.input_project = PointwiseLinear(in_channels, embed_dim)
    self.output_project = PointwiseLinear(embed_dim, out_channels)
    
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
    assert isinstance(domain_size_y, int)
    assert isinstance(domain_size_x, int)
    assert domain_size_y % self.sub_size_y == 0
    assert domain_size_x % self.sub_size_x == 0
    n_sub_y = domain_size_y // self.sub_size_y
    n_sub_x = domain_size_x // self.sub_size_x
    
    v = self.input_project(v)
    d = decompose2d(v, n_sub_x, n_sub_y)
    d = seq_op(self.padding.pad, d)
            
    for embed in self.embed:
      d = embed(d, n_sub_x, n_sub_y)
      
    d = seq_op(self.padding.unpad, d)
    u = recompose2d(d, n_sub_x, n_sub_y)
    u = self.output_project(u)
      
    return u