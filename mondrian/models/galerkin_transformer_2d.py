import math

import torch
from torch import nn
import einops

from mondrian.grid.attention.galerkin import galerkin_attention
from mondrian.grid.utility import grid
from .mlp import MLP

class GalerkinSelfAttention(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super().__init__()
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    
    # These are not packed because of the initialization.
    self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
    self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
    self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
    self._linear_init(self.wq.weight.data)
    self._linear_init(self.wk.weight.data)
    self._linear_init(self.wv.weight.data)

    self.ln_key = nn.LayerNorm(self.head_dim)
    self.ln_value = nn.LayerNorm(self.head_dim)
    
  def _linear_init(self, data):
    r"""
    This is the diagonal domaination weight initialization for Q, K, V weights. 
    This is described in section 5 of Galerkin Transormer, Shuhao Cao.
    The gain and delta are based on their ablation in the appendix.
    """
    with torch.no_grad():
      spread = math.sqrt(3 / self.embed_dim)
      data.uniform_(-spread, spread)
      gain, delta = 1e-2, 1e-2
      data *= gain
      diagonal_view = torch.diagonal(data) 
      diagonal_view += delta

  def forward(self, x):
    r"""
    Args:
      x: [batch, seq, embed]
    """
    rearrange_str = 'b s (heads dim) -> b heads s dim'
    query = einops.rearrange(self.wq(x), rearrange_str, heads=self.num_heads)
    key = einops.rearrange(self.wk(x), rearrange_str, heads=self.num_heads)
    value = einops.rearrange(self.wv(x), rearrange_str, heads=self.num_heads)

    key = self.ln_key(key)
    value = self.ln_value(value)
    ga_heads = galerkin_attention(query, key, value)
    ga = einops.rearrange(ga_heads, 'b heads s dim -> b s (heads dim)')
    return ga

class GalerkinEncoder(nn.Module):
  def __init__(self, 
               embed_dim, 
               num_heads):
    super().__init__()
    self.attn = GalerkinSelfAttention(embed_dim, num_heads)
    self.mlp = MLP(embed_dim, embed_dim, embed_dim, num_layers=2)
  
  def forward(self, x):
    x = self.attn(x) + x
    x = self.mlp(x) + x
    return x
    
class GalerkinTransformer2d(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               embed_dim,
               num_heads,
               num_layers):
    assert embed_dim % num_heads == 0
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    
    self.lift = MLP(in_channels + 2, embed_dim, embed_dim)
    self.project = MLP(embed_dim, out_channels, embed_dim, 2)
    
    self.encoders = nn.ModuleList([
      GalerkinEncoder(embed_dim, num_heads) for _ in range(num_layers)
    ])
  
  def forward(self, x, domain_size_x, domain_size_y):
    r"""
    The input is assumed to be a 2d grid. This rearranges it so the pixels can
    be interpreted as points.
    Args:
      x: [batch, channels, height, width]
    Returns:
      y: [batch, channels, height, width]
    """
    height = x.size(2)
    width = x.size(3)
    
    # add point-wise positions
    g = grid((x.size(-2), x.size(-1)), (domain_size_y, domain_size_x)).to(x.device)
    g = einops.repeat(g, '... -> b ...', b=x.size(0))
    x = torch.cat((g, x), dim=1)
    
    seq = einops.rearrange(x, 'b c h w -> b (h w) c')
    
    seq = self.lift(seq)
    for encoder in self.encoders:
      seq = encoder(seq)
    seq = self.project(seq)
    
    y = einops.rearrange(seq, 'b (h w) c -> b c h w', h=height, w=width)
    
    return y