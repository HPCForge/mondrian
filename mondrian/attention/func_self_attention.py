import functools
from typing import Optional

import einops
import torch
from torch import nn

from .functional.func_attention import func_attention
from ..layers.log_cpb import LogCPB
from ..grid.decompose import decompose2d, recompose2d
from ..grid.utility import is_power_of_2
from ..constants import HEAD_SPLIT_OPTIONS, CHANNEL, SPATIAL
from ..layers.qkv_operator import get_qkv_operator

# As implemented, this would be very slow to compute each iteration, since it's just loops.
# For bubbleml, the number of subdomains won't change, so cache it.
# torch.compile also takes forever to compile this, so it's disabled
@torch.compiler.disable
@functools.lru_cache(maxsize=16)
def neighborhood_mask(n_sub_x: int, n_sub_y: int, radius: int, device: torch.device):
    r"""
    This computes a mask for attention to only do attention between nearby subdomains.
    """
    mask = torch.zeros(n_sub_y, n_sub_x, n_sub_y, n_sub_x, dtype=bool, device=device)
    for i in range(n_sub_y):
        for j in range(n_sub_x):
            for r1 in range(-radius, radius + 1):
                for r2 in range(-radius, radius + 1):
                    # get index of neighboring cell
                    nbr_row_idx = i + r1
                    nbr_col_idx = j + r2 
                    # force neighborhoods to be inbounds 
                    nbr_row_idx = min(n_sub_y - 1, max(0, nbr_row_idx))
                    nbr_col_idx = min(n_sub_x - 1, max(0, nbr_col_idx))
                    mask[i, j, nbr_row_idx, nbr_col_idx] = True     
    mask = einops.rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask

class FuncSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        channel_heads: int,
        x_heads: int,
        y_heads: int,
        qkv_config: dict,
        attn_neighborhood_radius: Optional[torch.Tensor],
        use_bias: bool,
    ):
        super().__init__()
        num_heads = channel_heads * x_heads * y_heads
        assert is_power_of_2(num_heads)
        self.embed_dim = embed_dim
        self.channel_heads = channel_heads
        self.x_heads = x_heads
        self.y_heads = y_heads

        self.head_dim = embed_dim // channel_heads
        assert is_power_of_2(self.head_dim)
        
        # attn_mask can control the neighborhood attn is applied on
        # if it is None, just computes the full global attention.
        assert attn_neighborhood_radius is None or isinstance(attn_neighborhood_radius, int)
        self.attn_neighborhood_radius = attn_neighborhood_radius
        
        # TODO: this is only used for swin...
        self.use_bias = use_bias
        
        self.qkv_operator = get_qkv_operator(
            in_channels=embed_dim,
            out_channels=3 * embed_dim,
            bias=False,
            **qkv_config)
        self.output_operator = get_qkv_operator(
            in_channels=embed_dim, 
            out_channels=embed_dim, 
            bias=True,
            **qkv_config)

    def _qkv(self, v):
        dims = [1 for _ in range(v.dim())]
        dims[2] = 3
        v = self.qkv_operator(v)
        return v
    
    def _forward_spatial_heads(self, seq, n_sub_x, n_sub_y):
        r"""
        Computes multihead attention by partitioning subdomains spatially to create heads.
        """
        seq = self._qkv(seq)
        
        if self.x_heads > 1 or self.y_heads > 1:
            seq_heads = decompose2d(seq, self.x_heads, self.y_heads)
        else:
            seq_heads = einops.rearrange(seq, 'b s ... -> b s () ...')
            
        query, key, value = einops.rearrange(
            seq_heads, 
            'b s heads (split channel_heads c) h w -> split b (heads channel_heads) s c h w',
            split=3,
            channel_heads=self.channel_heads)
        
        if self.attn_neighborhood_radius is None:
            attn_mask = None
        else:
            attn_mask = neighborhood_mask(n_sub_x, n_sub_y, self.attn_neighborhood_radius, seq.device)
                    
        sa = func_attention(
            query, key, value, attn_mask
        )
        
        sa = einops.rearrange(
            sa, 
            'b (heads channel_heads) s c h w -> b s heads (channel_heads c) h w',
            channel_heads=self.channel_heads)
        sa = recompose2d(sa, self.x_heads, self.y_heads)
        
        return self.output_operator(sa)


    def forward(self, seq, n_sub_x, n_sub_y):
        r"""
        Args:
          seq: [batch, seq, channels, ...], a sequence of functions
          n_sub_x: number of subdomains in the x-direction
          n_sub_y: number of subdomains in the y-direction
        Returns:
          [batch, seq, channels, ...]
        """
        return self._forward_spatial_heads(seq, n_sub_x, n_sub_y)