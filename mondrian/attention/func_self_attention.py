import einops
import torch
from torch import nn

from .functional.func_attention import func_attention
from ..layers.log_cpb import LogCPB
from ..grid.decompose import decompose2d, recompose2d
from ..grid.utility import is_power_of_2
from ..constants import HEAD_SPLIT_OPTIONS, CHANNEL, SPATIAL
from ..layers.qkv_operator import get_default_qkv_operator

class FuncSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        channel_heads: int,
        x_heads: int,
        y_heads: int,
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
        self.use_bias = use_bias
        
        self.qkv_operator = get_default_qkv_operator(embed_dim, 3 * embed_dim, bias=False)
        self.output_operator = get_default_qkv_operator(embed_dim, embed_dim, bias=True)

    def _qkv(self, v):
        dims = [1 for _ in range(v.dim())]
        dims[2] = 3
        v = self.qkv_operator(v)
        return v

    def _forward_channel_heads(self, seq, n_sub_x, n_sub_y):
        r"""
        Computes the multihead by partitioning along the channels axis
        """
        with torch.profiler.record_function('compute qkv'):
            seq = self._qkv(seq)
        with torch.profiler.record_function('rearrange input'):
            heads = einops.rearrange(seq, 'b s (num_heads head_dim) ... -> b num_heads s head_dim ...', 
                                     num_heads=self.num_heads)
            query, key, value = einops.rearrange(heads, 'b h s (split d) ... -> split b h s d ...', split=3)

        with torch.profiler.record_function('func attention'):
            sa = func_attention(
                query.half(),
                key.half(),
                value.half()
            ).float()
            
        with torch.profiler.record_function('rearrange output'):
            sa = einops.rearrange(sa, "b h s d ... -> b s (h d) ...")
            
        return self.output_operator(sa)
    
    def _forward_spatial_heads(self, seq, n_sub_x, n_sub_y):
        r"""
        Computes multihead attention by partitioning subdomains spatially to create heads.
        """
        seq = self._qkv(seq)
        
        seq_heads = decompose2d(seq, self.x_heads, self.y_heads)
        query, key, value = einops.rearrange(
            seq_heads, 
            'b s heads (split channel_heads c) h w -> split b (heads channel_heads) s c h w',
            split=3,
            channel_heads=self.channel_heads)
        
        sa = func_attention(
            query.half(),
            key.half(),
            value.half()
        ).float()
        
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
        #if self.head_split == CHANNEL:
        #    return self._forward_channel_heads(seq, n_sub_x, n_sub_y)
        #if self.head_split == SPATIAL:
        return self._forward_spatial_heads(seq, n_sub_x, n_sub_y)