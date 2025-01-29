import math
from typing import List, Tuple

import torch

from .func_spda import func_spda_fa, func_spda

def func_attention(query, key, value, attn_mask=None):
    r"""
    Args:
        query: [batch, head, seq, channels, ...]
        key: [batch, head, seq, channels, ...]
        value: [batch, head, seq, channels, ...]
    """
    size = query.size()
    head_dim = math.prod(size[3:])
    
    #assert query.dtype in (torch.float16, torch.bfloat16)
    #assert head_dim <= 256
    #return func_spda_fa(query, key, value, attn_mask)
    return func_spda(query, key, value, attn_mask)