from typing import Tuple
import math

import einops
import torch
from torch import nn
from .functional.func_attention import func_attention
from ..grid.decompose import win_decompose2d, win_recompose2d
from ..layers.log_cpb import LogCPB
from ..grid.utility import is_power_of_2
from ..constants import HEAD_SPLIT_OPTIONS
from ..layers.qkv_operator import get_default_qkv_operator, get_qkv_operator

@torch.compile
class SwinFuncSelfAttention(nn.Module):
  def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_split: str,
        use_bias: bool,
        shift_size: int,
        n_sub_x: int,
        n_sub_y: int,
        window_size: int,
        qkv_config: dict,
    ):
      super().__init__()
      assert head_split in HEAD_SPLIT_OPTIONS
      assert is_power_of_2(num_heads)
      self.embed_dim = embed_dim
      self.num_heads = num_heads
      self.head_dim = embed_dim // num_heads
      assert is_power_of_2(self.head_dim)
      self.head_split = head_split
      self.use_bias = use_bias
      
      self.qkv_operator = get_qkv_operator(in_channels=embed_dim, out_channels=3 * embed_dim, bias=False, **qkv_config)
      self.output_operator = get_qkv_operator(in_channels=embed_dim, out_channels=embed_dim, bias=True, **qkv_config)
      
      if use_bias:
          self.log_cpb = LogCPB(embed_dim, num_heads)
      else:
          self.log_cpb = None

      self.window_size = window_size
      self.shift_size = shift_size
      self.n_sub_x = n_sub_x
      self.n_sub_y = n_sub_y

      if self.shift_size > 0:
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, 1, self.n_sub_y, self.n_sub_x))
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

        mask_windows = win_decompose2d(img_mask, self.n_sub_y, self.n_sub_x, self.window_size)
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
  
  def _qkv(self, v):
        dims = [1 for _ in range(v.dim())]
        dims[2] = 3
        v = self.qkv_operator(v)
        return v

  def _shifted_forward_channel_heads(self, seq, n_sub_x, n_sub_y):
    r"""
    Computes the multihead by partitioning along the channels axis
    """
    subdomain_size_x = seq.size(-1)
    subdomain_size_y = seq.size(-2)
  
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
    
    if self.attn_mask is None and self.use_bias:
      bias = self.log_cpb(self.window_size, self.window_size, device=query.device)
    elif self.attn_mask is not None:
      # NOTE: the attn_mask can be different in different windows. So, the bias is computed for each window. 
      # When attention is called, the batch and window dimensions are merged, so we need to repeat the bias to account for this.
      bias = self.attn_mask + self.log_cpb(self.window_size, self.window_size, device=query.device) if self.log_cpb is not None else self.attn_mask
      # query already has batch and window merged, so this is safe.
      n_window = query.size(0) // self.attn_mask.size(0)
      ones = [1 for _ in range(bias.dim() - 1)]
      bias = bias.repeat(n_window, *ones)
    else:
      bias = None
      
    sa = func_attention(
        query,
        key,
        value,
        attn_mask=bias
    )    
    sa = einops.rearrange(sa, "b h s d ... -> b s (h d) ...")
    sa = self.output_operator(sa)
    
    if self.shift_size > 0:
      sa = win_recompose2d(sa, n_sub_x, n_sub_y, self.window_size)
      sa = torch.roll(sa, shifts=(self.shift_size * subdomain_size_x, self.shift_size * subdomain_size_y), dims=(-2, -1))
      sa = win_decompose2d(sa, n_sub_x, n_sub_y, self.window_size)

    return  sa
  
  def forward(self, seq, n_sub_x, n_sub_y):
    return self._shifted_forward_channel_heads(seq, n_sub_x, n_sub_y)