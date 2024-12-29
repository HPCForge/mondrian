from typing import List, Tuple

import torch

from mondrian.grid.attention.functional._naive import naive_func_attention
from .func_spda import func_spda


def func_attention(query, key, value, quadrature_weights, bias=None, return_scores=False):
    if quadrature_weights is None:
        quadrature_weights = 1
    if bias is not None:
        return naive_func_attention(query, key, value, quadrature_weights, bias=bias)
    return func_spda(query, key, value, quadrature_weights)
