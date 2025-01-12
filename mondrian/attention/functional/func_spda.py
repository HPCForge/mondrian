import math
import warnings

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...grid.quadrature import get_unit_quadrature_weights


@torch.compile
def func_spda_fa(query, key, value):
    r"""
    computes softmax(QWK^T)V.  The application of the quadrature weights is not fused.
    Flattening the channel and spatial dims is safe because they just get reduced.
    """
    # flash and cudnn attention both require inputs to be half precision or bfloat16.
    # At such low precisions, the quadrature method shouldn't really matter.
    # flash attention can use a head dimension of 256, but cudnn attention can only use 128.
    assert query.dtype in (torch.float16, torch.bfloat16)
    
    discretization = torch.prod(query.size()[4:])

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return scaled_dot_product_attention(
            query.flatten(start_dim=3),
            key.flatten(start_dim=3),
            value.flatten(start_dim=3),
            # `query.size(3)` is for normalizing variance (like original transformer paper)
            # This method does not apply quadrature weights to the query, since higher order methods
            # seem to not matter at lower precisions. So `discrtization` is for the interpretation as an integral.
            scale=1 / (math.sqrt(query.size(3)) * discretization),
        ).reshape(query.size())
        
@torch.compile
def func_spda(query, key, value):
    r"""
    computes softmax(QWK^T)V.  The application of the quadrature weights is not fused.
    Flattening the channel and spatial dims is safe because they just get reduced.
    """
    
    # This is memory bound, but could be manually fused into an attention kernel.
    # From asking on gpu-mode discord, none of torch's attention implementations
    # can fuse this out of the box. 
    height = query.size(-2)
    width = query.size(-1)
    quadrature_weights = get_unit_quadrature_weights((height, width), device=query.device)
    query = query * quadrature_weights
    
    return scaled_dot_product_attention(
        query.flatten(start_dim=3),
        key.flatten(start_dim=3),
        value.flatten(start_dim=3),
        # since we already apply the quadrature weights, we don't normalize by discretization
        scale=1 / math.sqrt(query.size(3)),
    ).reshape(query.size())