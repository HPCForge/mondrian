from typing import Union, Tuple

import einops
import torch
from torch import nn

from mondrian.grid.decompose import win_decompose2d, win_recompose2d
from mondrian.attention.swin_func_self_attention import SwinFuncSelfAttention
from mondrian.layers.seq_op import seq_op
from mondrian.layers.learned_pos_embedding import LearnedPosEmbedding2d
from mondrian.layers.feed_forward_operator import get_default_feed_forward_operator, get_feed_forward_operator


class SequenceGroupNorm2d(nn.Module):
    def __init__(self, num_groups, embed_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, embed_dim)

    def forward(self, v):
        return seq_op(self.norm, v)

class Encoder(nn.Module):
  def __init__(self, embed_dim, num_heads, head_split, use_bias, shift_size, n_sub_x, n_sub_y, window_size, ff_config, qkv_config):
    super().__init__()
    self.shift_size = shift_size
    self.sa = SwinFuncSelfAttention(
      embed_dim, num_heads, head_split, use_bias, shift_size, n_sub_x, n_sub_y, window_size, qkv_config
    )

    # One option is to use FNO, but that seems to work really poorly...
    self.mlp = get_feed_forward_operator(in_channels=embed_dim, out_channels=embed_dim, hidden_channels=embed_dim, **ff_config)
    self.norm1 = SequenceGroupNorm2d(8, embed_dim)
    self.norm2 = SequenceGroupNorm2d(8, embed_dim)
    
  def forward(self, v, n_sub_x, n_sub_y):
      with torch.profiler.record_function("self_attention"):
          v = self.sa(self.norm1(v), n_sub_x, n_sub_y) + v
      with torch.profiler.record_function("mlp"):
          v = self.mlp(self.norm2(v)) + v
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
    window_size: The number of subdomains in each window (assuming square window).
    shift_size: The number of subdomains to shift.
    n_sub: The number of subdomains the global domain.
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
               num_layers: int,
               window_size: int,
               shift_size: int,
               n_sub_x: int,
               n_sub_y: int,
               subdomain_size: Union[int, Tuple[int, int]],
               qkv_config: dict,
               ff_config: dict
               ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    if isinstance(subdomain_size, int):
      subdomain_size = (subdomain_size, subdomain_size)
    assert isinstance(subdomain_size[0], int) 
    assert isinstance(subdomain_size[1], int)

    self.n_sub_x = n_sub_x
    self.n_sub_y = n_sub_y
    self.shift_size = shift_size
    self.window_size = window_size
    self.subdomain_size = subdomain_size
    self.sub_size_y = self.subdomain_size[0]
    self.sub_size_x = self.subdomain_size[1]
    
    self.encoder = nn.ModuleList([
      Encoder(embed_dim, 
              num_heads, 
              head_split,
              True, 
              shift_size=0 if (i%2==0) else self.shift_size , 
              n_sub_x=self.n_sub_x,
              n_sub_y=self.n_sub_y, 
              window_size=self.window_size,
              ff_config=ff_config,
              qkv_config=qkv_config,) 
      for i in range(num_layers)
    ])
    
    self.input_project = get_feed_forward_operator(in_channels=in_channels, out_channels=embed_dim, hidden_channels=embed_dim, **ff_config)
    self.output_project = get_feed_forward_operator(in_channels=embed_dim, out_channels=out_channels, hidden_channels=embed_dim, **ff_config)
    self.pos_embedding = LearnedPosEmbedding2d(
            seq_len=window_size**2, channels=embed_dim
        )
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

    d = win_decompose2d(v, n_sub_x, n_sub_y, self.window_size)
    d = self.input_project(d)
    d = self.pos_embedding(d)

    for encoder in self.encoder:
      d = encoder(d, n_sub_x, n_sub_y)
            
    u = self.output_project(d)
    u = win_recompose2d(u, n_sub_x, n_sub_y, self.window_size)

    return u
  