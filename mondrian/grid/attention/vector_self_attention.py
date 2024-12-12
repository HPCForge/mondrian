import torch
from torch import nn
import einops

class VectorSelfAttention(nn.Module):
  def __init__(
    self, 
    embed_dim,
    num_heads
  ):
    self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
    self.mha = nn.MultiheadAttention(
      embed_dim, 
      num_heads, 
      batch_first=False)
    
  def _qkv(self, x):
    return einops.rearrange(self.qkv(x), 'b s (three e) -> three b s e', three=3)
  
  def forward(self, x):
    r"""
    Args:
      x: [batch, seq, embed]
    Returns:
      [batch, seq, embed]
    """
    query, key, value = self._qkv(x)
    # specifying need_weights=False enables using the optimized scaled_dot_product_attenion
    return self.mha(query, key, value, need_weights=False)