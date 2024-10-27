import torch
from torch import nn

from mondrian.grid.decompose import decompose2d, recompose2d
from mondrian.grid.spectral_conv import SimpleSpectralConv
from mondrian.grid.vit_self_attention_operator import ViTSelfAttentionOperator

class Embed(nn.Module):
  def __init__(self, 
               embed_dim, 
               num_heads):
    super().__init__()
    self.sa = ViTSelfAttentionOperator(embed_dim, num_heads)
    modes = (8, 8)
    self.spectral_conv = SimpleSpectralConv(embed_dim, embed_dim, modes)
    
  def forward(self, v):
    # TODO: figure out decent way to do normalization...
    v = (self.sa(v)) + v
    v = (self.spectral_conv(v)) + v
    return nn.functional.gelu(v)

class ViTOperator(nn.Module):
  def __init__(self,
               embed_dim,
               num_heads,
               num_layers,
               subdomain_size: int):
    super().__init__()
    self.subdomain_size = subdomain_size
    self.embed = nn.ModuleList([
      Embed(embed_dim, num_heads) for _ in range(num_layers)
    ])
  
  def forward(self, v: torch.Tensor, domain_size_y: int, domain_size_x: int):
    assert isinstance(domain_size_y, int)
    assert isinstance(domain_size_x, int)
    assert domain_size_y % self.subdomain_size == 0
    assert domain_size_x % self.subdomain_size == 0
    n_sub_y = domain_size_y // self.subdomain_size
    n_sub_x = domain_size_x // self.subdomain_size
    
    d = decompose2d(v, n_sub_x, n_sub_y)
        
    for embed in self.embed:
      d = embed(d)
      
    u = recompose2d(d, n_sub_x, n_sub_y)
      
    return u