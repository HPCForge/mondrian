import math
from typing import Optional
import einops
import torch
from torch.nn.functional import softmax
from mondrian.grid.quadrature import integrate


def left_reimann_1d(f, dx):
    return (f[..., :-1] * dx).sum(dim=-1)


def inner_product_score(query, key, quadrature_weights):
    r"""
    Computes and inner product based score. The quadrature weights
    should be selected to mimic an integration method and be broadcastable
    with the query's discretization.
    Args:
      query: [batch x heads x seq1 x channels x ...]
      key: [batch x heads x seq2 x channels x ...]
      quadrature_weights: [...]
    """
    return einops.einsum(
        query * quadrature_weights, key, "b h s1 ..., b h s2 ... -> b h s1 s2"
    )


def naive_func_attention(query, key, value, quadrature_weights, bias=None):
    score = inner_product_score(query, key, quadrature_weights)
    if bias is not None:
        score = score + bias
    probabilities = softmax(score.float(), dim=-1)
    return einops.einsum(probabilities, value, "b h s1 s2, b h s2 ... -> b h s1 ...")
