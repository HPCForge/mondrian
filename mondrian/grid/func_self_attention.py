from typing import Tuple
import math

import einops
import torch
from torch import nn

from .fast_math import attention
from .spectral_conv import SimpleSpectralConv2d
from .log_cpb import LogCPB
from .decompose import decompose2d, recompose2d, win_decompose2d, win_recompose2d
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
                shift_size: int,
                n_sub: int,
                window_size: int):
      super().__init__(embed_dim, num_heads, 'channel', use_bias, 'reimann')
      self.window_size = window_size
      self.shift_size = shift_size
      self.n_sub = n_sub

      if self.shift_size > 0:
        # calculate attention mask for SW-MSA
        H, W = n_sub, n_sub
        img_mask = torch.zeros((1, 1, H, W))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:,:, h, w] = cnt
                cnt += 1

        mask_windows = win_decompose2d(img_mask, H, W, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0))
        
        dims = [1 for _ in range(attn_mask.dim()+1)]
        dims[0] = self.num_heads
        attn_mask = attn_mask.unsqueeze(0).repeat(dims)
        # nW x nheads x win_size^2 x win_size^2
        attn_mask = attn_mask.contiguous().permute(1,0,2,3)
      else:
        attn_mask = None

      self.register_buffer("attn_mask", attn_mask)
  
  def _shifted_forward_channel_heads(self, seq, n_sub_x, n_sub_y):
    r"""
    Computes the multihead by partitioning along the channels axis
    """
    subdomain_size_x = seq.size(-1) // n_sub_x
    subdomain_size_y = seq.size(-2) // n_sub_y

    if self.shift_size > 0:
      seq = win_recompose2d(seq, n_sub_x, n_sub_y, self.window_size)
      seq = torch.roll(seq, shifts=(-self.shift_size * subdomain_size_x, -self.shift_size * subdomain_size_y), dims=(-2, -1))
      seq = win_decompose2d(seq, n_sub_x, n_sub_y, self.window_size)

    query, key, value = einops.rearrange(
        self._qkv(seq),
        'b s (qkv num_heads head_dim) ... -> qkv b num_heads s head_dim ...',
        qkv=3,
        num_heads=self.num_heads,
        head_dim=self.head_dim)
    
    if self.attn_mask is None:
      bias = self.log_cpb(n_sub_x, n_sub_y, device=query.device)
    else:
      bias = self.attn_mask + self.log_cpb(n_sub_x, n_sub_y, device=query.device) if self.log_cpb is not None else self.attn_mask

    sa = attention(query, key, value, bias=bias, shifted=True, score_method=self.score_method)
    sa = self._flatten(sa)
    sa = self.output_operator(sa)
    
    
    if self.shift_size > 0:
      sa = win_recompose2d(sa, n_sub_x, n_sub_y, self.window_size)
      sa = torch.roll(sa, shifts=(self.shift_size * subdomain_size_x, self.shift_size * subdomain_size_y), dims=(-2, -1))
      sa = win_decompose2d(sa, n_sub_x, n_sub_y, self.window_size)

    return  sa
  
  def forward(self, seq):
    return self._shifted_forward_channel_heads(seq, self.window_size, self.window_size)

