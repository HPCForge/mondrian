import torch
import einops


def galerkin_attention(query, key, value):
    r"""
    Implements the Galerkin-style attention operation: `https://arxiv.org/pdf/2105.14995`
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
    weights = einops.einsum(key, value, "... s_k e_k, ... s_k e_v -> ... e_k e_v")
    seq_q = query.size(-2)
    return (
        einops.einsum(query, weights, "... s_q e_k, ... e_k e_v -> ... s_q e_v") / seq_q
    )
