from typing import Union, Tuple

import torch
from torch import nn

from mondrian.grid.decompose import decompose2d, recompose2d
from mondrian.grid.spectral_conv import SimpleSpectralConv
from mondrian.grid.vit_self_attention_operator import ViTSelfAttentionOperator
from mondrian.grid.pointwise_linear import PointwiseLinear

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
    batch_size = v.size(0)
    seq_len = v.size(1)
    v = torch.flatten(v, start_dim=0, end_dim=1)
    v = self.norm(v)
    v = torch.unflatten(v, dim=0, sizes=(batch_size, seq_len))
    return v

class Embed(nn.Module):
  def __init__(self, 
               embed_dim, 
               num_heads):
    super().__init__()
    self.sa = ViTSelfAttentionOperator(embed_dim, num_heads)
    modes = (16, 16)
    self.spectral_conv1 = SimpleSpectralConv(embed_dim, embed_dim, modes)
    self.spectral_conv2 = SimpleSpectralConv(embed_dim, embed_dim, modes)

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
      Embed(embed_dim, num_heads) for _ in range(num_layers)
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
    batch_size = d.size(0)
    seq_len = d.size(1)
    d = self.flatten(d)
    d = self.padding.pad(d)
    d = self.unflatten(d, batch_size, seq_len)
        
    for embed in self.embed:
      d = embed(d, n_sub_x, n_sub_y)
      
    d = self.flatten(d)
    d = self.padding.unpad(d)
    d = self.unflatten(d, batch_size, seq_len)
    u = recompose2d(d, n_sub_x, n_sub_y)
    u = self.output_project(u)
      
    return u