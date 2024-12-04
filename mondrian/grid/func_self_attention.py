from typing import Tuple
import math

import einops
import torch
from torch import nn

from .fast_math import attention
from .spectral_conv import SimpleSpectralConv2d
from .log_cpb import LogCPB
from .decompose import decompose2d, recompose2d
from .utility import is_power_of_2

CHANNEL = 'channel'
SPATIAL = 'spatial'
HEAD_SPLIT_OPTIONS = [
  CHANNEL, SPATIAL
]

class FuncSelfAttention(nn.Module):
  def __init__(self,
               embed_dim: int,
               num_heads: int,
               head_split: str,
               use_bias: bool,
               score_method: str):
    super().__init__()
    assert head_split in HEAD_SPLIT_OPTIONS
    assert is_power_of_2(num_heads)
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    assert is_power_of_2(self.head_dim)
    self.head_split = head_split
    self.use_bias = use_bias
    self.score_method = score_method

    modes = 16
    self.qkv_operator = SimpleSpectralConv2d(embed_dim, 3 * embed_dim, modes)
    self.output_operator = SimpleSpectralConv2d(embed_dim, embed_dim, modes)
    if use_bias:
      self.log_cpb = LogCPB(embed_dim, num_heads) 
    else:
      self.log_cpb = None
    
  def _qkv(self, v):
    dims = [1 for _ in range(v.dim())]
    dims[2] = 3
    v = self.qkv_operator(v) + v.repeat(dims)
    return v
  
  def _flatten(self, f):
    # [batch x seq x heads x embed_dim x ...]
    t = torch.transpose(f, 1, 2)
    # [batch x seq x (heads x embed_dim)]
    return torch.flatten(t, start_dim=2, end_dim=3)
  
  def _forward_channel_heads(self, seq, n_sub_x, n_sub_y):
    r"""
    Computes the multihead by partitioning along the channels axis
    """
    query, key, value = einops.rearrange(
        self._qkv(seq), 
        'b s (qkv num_heads head_dim) ... -> qkv b num_heads s head_dim ...',
        qkv=3,
        num_heads=self.num_heads,
        head_dim=self.head_dim)
    
    if self.log_cpb is not None:
      bias = self.log_cpb(n_sub_x, n_sub_y, device=query.device)
    else:
      bias = None
      
    sa = attention(query, key, value, bias=bias, score_method=self.score_method)
    sa = self._flatten(sa)
    
    return self.output_operator(sa)

  def _forward_function_heads(self, seq, n_sub_x, n_sub_y):
    r"""
    Computes multihead by partitioning the spatial components of the function
    """
    heads_x = int(math.sqrt(self.num_heads))  
    heads_y = int(math.sqrt(self.num_heads))
    
    batch_size = seq.size(0)
    seq_len = seq.size(1)
    
    # [batch_size x seq_len x (3 x embed_dim) x ...]
    qkv = self._qkv(seq)
    # View: [(batch_size x seq_len) x (3 x embed_dim) x ...]
    qkv_flat = torch.flatten(qkv, start_dim=0, end_dim=1)
    # [(batch_size x seq_len) x heads x (3 x embed_dim) x ...]
    qkv_heads = decompose2d(qkv_flat, heads_x, heads_y)
    
    query, key, value = einops.rearrange(
      qkv_heads,
      '(b s) num_heads (qkv e) ... -> qkv b num_heads s e ...',
      b=batch_size,
      s=seq_len,
      qkv=3)
    
    sa = attention(query, key, value, bias=None, score_method=self.score_method)    
    
    # recompose subdomains
    sa = einops.rearrange(sa, 'b h s ... -> (b s) h ...')
    sa = recompose2d(sa, heads_x, heads_y)
    sa = einops.rearrange(sa, '(b s) e ... -> b s e ...', b=batch_size, s=seq_len)
    
    return self.output_operator(sa)
    
  def forward(self, seq, n_sub_x, n_sub_y):
    if self.head_split == CHANNEL:
      return self._forward_channel_heads(seq, n_sub_x, n_sub_y)
    if self.head_split == SPATIAL:
      return self._forward_function_heads(seq, n_sub_x, n_sub_y)
  
class WinFuncSelfAttention(FuncSelfAttention):
  def __init__(self,
                embed_dim: int,
                num_heads: int,
                use_bias: bool,
                window_size: int):
      super().__init__(embed_dim, num_heads, use_bias)
      self.window_size = window_size
      
  def forward(self, seq):
    return self._forward_channel_heads(seq, self.window_size, self.window_size)

