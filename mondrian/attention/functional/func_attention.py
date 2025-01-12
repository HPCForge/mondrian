from typing import List, Tuple

import torch

from .func_spda import func_spda_fa, func_spda

def func_attention(query, key, value):
    r"""
    Args:
        query: [batch, head, seq, channels, ...]
        key: [batch, head, seq, channels, ...]
        value: [batch, head, seq, channels, ...]
    """
    size = query.size()
    head_dim = size[3] * size[4] * size[5]
    
    if query.dtype in (torch.float16, torch.bfloat16) and head_dim <= 256:
        return func_spda_fa(query, key, value)
    return func_spda(query, key, value)