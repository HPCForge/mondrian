import math
import warnings

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

def func_spda_fa(query, key, value, attn_mask):
    r"""
    computes softmax(QWK^T)V.  The application of the quadrature weights is not fused.
    Flattening the channel and spatial dims is safe because they just get reduced.
    """
    # flash and cudnn attention both require inputs to be half precision or bfloat16.
    # At such low precisions, the quadrature method shouldn't really matter.
    # flash attention can use a head dimension of 256, cudnn attention can only use 128.
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert math.prod(query.size()[3:]) <= 256

    height = query.size(-2)
    width = query.size(-1)
    
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return scaled_dot_product_attention(
            # TODO: for some reason these don't have stride 1? So need .contiguous
            query.flatten(start_dim=3).contiguous(),
            key.flatten(start_dim=3).contiguous(),
            value.flatten(start_dim=3).contiguous(),
            attn_mask=attn_mask,
            # `query.size(3)` is for normalizing variance (like original transformer paper)
            # height * width is for interpretation as integral
            scale=1 / (math.sqrt(query.size(3)) * height * width)
        ).reshape(query.size())
        
def func_spda(query, key, value, attn_mask):
    r"""
    computes softmax(QWK^T)V.  The application of the quadrature weights is not fused.
    Flattening the channel and spatial dims is safe because they just get reduced.
    """
    
    height = query.size(-2)
    width = query.size(-1)
    
    return scaled_dot_product_attention(
        query.flatten(start_dim=3),
        key.flatten(start_dim=3),
        value.flatten(start_dim=3),
        attn_mask=attn_mask,
        scale=1 / (math.sqrt(query.size(3)) * height * width),
    ).reshape(query.size())