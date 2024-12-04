from typing import Tuple
import math

import torch
from torch import nn

from .fast_math import attention, is_power_of_2
from .spectral_conv import SimpleSpectralConv2d
from .log_cpb import LogCPB
from .decompose import decompose2d, recompose2d

class FuncSelfAttention(nn.Module):
  def __init__(self,
               embed_dim: int,
               num_heads: int,
               use_bias: bool):
    super().__init__()
    assert is_power_of_2(embed_dim)
    assert is_power_of_2(num_heads)
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.use_bias = use_bias

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
    # [3 x batch_size x heads x seq_len x head_dim x ...]
    qkv = self._qkv(seq) \
      .unflatten(dim=2, sizes=(3, self.num_heads, self.head_dim)) \
      .movedim(source=2, destination=0) \
      .movedim(source=3, destination=2)
    query, key, value = qkv[0], qkv[1], qkv[2]
    
    if self.log_cpb is not None:
      bias = self.log_cpb(n_sub_x, n_sub_y, device=query.device)
    else:
      bias = None
    sa = attention(query, key, value, bias)
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
    # pack the operator to compute q, k, and v.
    # [batch_size x seq_len x (3 x embed_dim) x ...]
    qkv = self._qkv(seq)
    qkv_flat = torch.flatten(qkv, start_dim=0, end_dim=1)
    # [(batch_size x seq_len) x heads x (3 x embed_dim) x ...]
    qkv_heads = decompose2d(qkv_flat, heads_x, heads_y)
    # [batch_size x seq_len x heads x (3 x embed_dim) x ...]
    qkv_heads = torch.unflatten(qkv_heads, dim=0, sizes=(batch_size, seq_len))
    qkv_heads = torch.unflatten(qkv_heads, dim=3, sizes=(3, self.embed_dim))
    # [3 x batch_size x heads x seq_len x embed_dim x ...]
    qkv_heads = qkv_heads.transpose(1, 2) \
      .movedim(source=3, destination=0)
    query, key, value = qkv_heads[0], qkv_heads[1], qkv_heads[2]

    #TODO: figure out better position setup...
    bias = self.log_cpb(n_sub_x, n_sub_y, device=query.device)
    sa = attention(query, key, value, bias)
    
    # merge batch and seq dims.
    sa = sa.transpose(1, 2)
    sa_flat = torch.flatten(sa, start_dim=0, end_dim=1)
    
    # recompose subdomains
    sa_flat = recompose2d(sa_flat, heads_x, heads_y)
    sa = torch.unflatten(sa_flat, dim=0, sizes=(batch_size, seq_len))
    
    return self.output_operator(sa)
    
  def forward(self, seq, n_sub_x, n_sub_y):
    return self._forward_channel_heads(seq, n_sub_x, n_sub_y)
  
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