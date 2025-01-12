import einops
import torch
from torch import nn

from .functional.func_attention import func_attention
from ..layers.log_cpb import LogCPB
from ..grid.utility import is_power_of_2
from ..constants import HEAD_SPLIT_OPTIONS
from ..layers.qkv_operator import get_default_qkv_operator

class FuncSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_split: str,
        use_bias: bool,
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
        
        self.qkv_operator = get_default_qkv_operator(embed_dim, 3 * embed_dim, bias=False)
        self.output_operator = get_default_qkv_operator(embed_dim, embed_dim, bias=True)
        
        if use_bias:
            self.log_cpb = LogCPB(embed_dim, num_heads)
        else:
            self.log_cpb = None

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
                query,
                key,
                value
            )
            
        with torch.profiler.record_function('rearrange output'):
            sa = einops.rearrange(sa, "b h s d ... -> b s (h d) ...")
            
        return self.output_operator(sa)

    def forward(self, seq, n_sub_x, n_sub_y):
        r"""
        Args:
          seq: [batch, seq, channels, ...], a sequence of functions
          n_sub_x: number of subdomains in the x-direction
          n_sub_y: number of subdomains in the y-direction
        Returns:4
          [batch, seq, channels, ...]
        """
        return self._forward_channel_heads(seq, n_sub_x, n_sub_y)
