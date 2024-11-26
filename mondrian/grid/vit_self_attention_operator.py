from typing import Tuple

import torch
from torch import nn

from .fast_math import self_attention
from .spectral_conv import SimpleSpectralConv
from .log_cpb import LogCPB

class ViTSelfAttentionOperator(nn.Module):
  r"""
  This is a ViT-style softmax self-attention.
  The inspiration from ViT is that this is applied to
  a sequence of functions, in a similar style to how ViT
  is applied to a sequence of image patches.
  """
  def __init__(self,
               embed_dim,
               num_heads):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads

    modes = (16, 16)
    self.query_operator = SimpleSpectralConv(embed_dim, embed_dim * num_heads, modes)
    self.key_oeprator = SimpleSpectralConv(embed_dim, embed_dim * num_heads, modes)
    self.value_operator = SimpleSpectralConv(embed_dim, embed_dim * num_heads, modes)
    self.output_operator = SimpleSpectralConv(num_heads * embed_dim, embed_dim, modes)
    self.log_cpb = LogCPB(embed_dim, num_heads)

  def _unflatten(self, f):
    f = torch.unflatten(f, 2, (self.num_heads, self.embed_dim))
    return torch.transpose(f, 1, 2)
  
  def _flatten(self, f):
    # [batch x seq x heads x embed_dim x ...]
    t = torch.transpose(f, 1, 2)
    # [batch x seq x (heads x embed_dim)]
    return torch.flatten(t, start_dim=2, end_dim=3)
  
  def forward(self, seq, n_sub_x, n_sub_y):
    r"""
    Args:
      seq: [batch_size x seq_len x embed_dim x ...]
    """
    assert seq.dim() > 3
        
    # [batch_size x seq_len x (embed_dim x heads) x ...]
    query = self.query_operator(seq)
    key = self.key_oeprator(seq)
    value = self.value_operator(seq)
    
    # [batch_size x heads x seq_len x embed_dim x ...]
    query = self._unflatten(query)
    key = self._unflatten(key)
    value = self._unflatten(value)
    
    bias = self.log_cpb(n_sub_x, n_sub_y, device=query.device)
    
    sa = self_attention(query, key, value, bias)
    sa = self._flatten(sa)
    
    return self.output_operator(sa)
  
class WinSelfAttentionOperator(nn.Module):
  r"""
  The is the implementation of a single window self-attention operation. 
  """
  def __init__(self,
               embed_dim,
               num_heads,
               window_size: int):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.window_size = window_size

    modes = (16, 16)
    self.query_operator = SimpleSpectralConv(embed_dim, embed_dim * num_heads, modes)
    self.key_oeprator = SimpleSpectralConv(embed_dim, embed_dim * num_heads, modes)
    self.value_operator = SimpleSpectralConv(embed_dim, embed_dim * num_heads, modes)
    self.output_operator = SimpleSpectralConv(num_heads * embed_dim, embed_dim, modes)
    self.log_cpb = LogCPB(embed_dim, num_heads)

  def _unflatten(self, f):
    f = torch.unflatten(f, 2, (self.num_heads, self.embed_dim))
    return torch.transpose(f, 1, 2)
  
  def _flatten(self, f):
    # [batch x seq x heads x embed_dim x ...]
    t = torch.transpose(f, 1, 2)
    # [batch x seq x (heads x embed_dim)]
    return torch.flatten(t, start_dim=2, end_dim=3)
  
  def forward(self, seq):
    r"""
    Args:
      seq: [batch_size x seq_len x embed_dim x ...]
    """
    assert seq.dim() > 3
        
    # [batch_size x seq_len x (embed_dim x heads) x ...]
    query = self.query_operator(seq)
    key = self.key_oeprator(seq)
    value = self.value_operator(seq)
    
    # [batch_size x heads x seq_len x embed_dim x ...]
    query = self._unflatten(query)
    key = self._unflatten(key)
    value = self._unflatten(value)
    
    bias = self.log_cpb(self.window_size, self.window_size, device=query.device)
    
    sa = self_attention(query, key, value, bias)
    sa = self._flatten(sa)
    
    return self.output_operator(sa)