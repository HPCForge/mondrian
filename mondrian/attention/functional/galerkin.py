from typing import Union
import torch
import einops

from ...grid.quadrature import get_unit_quadrature_weights


@torch.compile
def galerkin_attention(query, key, value, quadrature_weights):
    r"""
    Implements the Galerkin-style attention operation:
      1. https://arxiv.org/pdf/2105.14995
      2. https://arxiv.org/abs/1812.01243
    This essentially just changes `softmax(QK^T)V to Q(K^TV), which reduces the
    complexity to be linear in the sequence length, since K^TV will be much smaller than QK^T.

    This assumes that the caller has already applied some normalization to the key and value.
    The paper uses layer normalization.

    Args:
      query: [..., s_q, e_k]
      key: [..., s_k, e_k]
      value: [..., s_k, e_v]
    Returns:
      [..., s_q, e_v]
    """
    weights = torch.matmul(key.transpose(-2, -1) * quadrature_weights, value)
    return torch.matmul(query, weights)