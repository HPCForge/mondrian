import math
from typing import Optional
import einops
import torch
from torch.nn.functional import softmax


def inner_product_score(query, key, quadrature_weights):
    r"""
    Computes an inner product based score. The quadrature weights
    should be selected to mimic an integration method and be broadcastable
    with the query's discretization.

    This is essentially computing QWK^T, where W is a diagonal matrix with the
    quadrature weights, which is intended to mimic the inner product of a query and key function.
    "transposing" K essentially reverses all of the axes
    Once the quadrature weights have been applied, the function discretizations can be
    flattened, which is why it's done with a GEMM.

    Args:
      query: [batch x heads x seq1 x channels x ...]
      key: [batch x heads x seq2 x channels x ...]
      quadrature_weights: [...]
    """
    normalization = math.sqrt(query.size(3))
    return (
        einops.einsum(
            query * quadrature_weights, key, "b h s1 ..., b h s2 ... -> b h s1 s2"
        )
        / normalization
    )


def naive_func_attention(
    query, key, value, quadrature_weights, bias=None, shifted=False, return_scores=False
):
    score = inner_product_score(query, key, quadrature_weights)
    if bias is not None:
        if not shifted:
            score = score + bias
        else:
            # nWindows is flattened in the batch dimension
            # batch*nWindows x nHeads x nSeq1 x nSeq2 -> batch x nWindows x nHeads x nSeq1 x nSeq2
            batch_size = score.size(0)
            num_windows = bias.size(0)
            score = score.unflatten(0, (batch_size//num_windows, num_windows)) + bias
            score = score.flatten(0, 1)
    probabilities = softmax(score, dim=-1)
    output = einops.einsum(probabilities, value, "b h s1 s2, b h s2 ... -> b h s1 ...")
    if return_scores:
        return output, score
    return output
