from typing import List, Tuple

import torch

from mondrian.grid.attention.functional._naive import naive_func_attention


def func_attention(
    query,
    key,
    value,
    quadrature_weights,
    bias=None,
):
    return naive_func_attention(query, key, value, quadrature_weights, bias)
