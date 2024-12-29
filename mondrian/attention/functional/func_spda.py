import math

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


def func_spda(query, key, value, quadrature_weights):
    r"""
    computes softmax(QWK^T)V.  The application of the quadrature weights is not fused.
    Flattening the channel and spatial dims is safe because they just get reduced.
    """
    return scaled_dot_product_attention(
        (query * quadrature_weights).flatten(start_dim=3),
        key.flatten(start_dim=3),
        value.flatten(start_dim=3),
        scale=1 / math.sqrt(query.size(3)),
    ).reshape(query.size())
