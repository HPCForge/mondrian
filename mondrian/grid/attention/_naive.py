import math
from typing import Optional
import einops
import torch
from torch.nn.functional import softmax

SCORE_METHODS = ['reimann', 'trapezoid']

def left_reimann_1d(f, dx):
  return (f[..., :-1] * dx).sum(dim=-1)

def reimann_inner_product_score(u, v, p=2):
  r"""
  Computes and inner product based score. This
  approximates the integral with a reimann sum.
  Args:
    u: [batch x heads x seq1 x channels x ...]
    v: [batch x heads x seq2 x channels x ...]
  """
  assert p == 2
  local_product = einops.einsum(
    u, v, 'b h s1 ..., b h s2 ... -> b h s1 s2 ...')
  local_inner_product = local_product.sum(dim=4)
  numel = local_inner_product[0,0,0,0].numel()
  assert numel > 0
  dims_to_sum = list(range(4, local_inner_product.ndim))
  if not dims_to_sum:
    return local_inner_product
  integral = local_inner_product
  for _ in range(len(dims_to_sum)):
    integral = left_reimann_1d(integral, dx=1/integral.size(-1))
  return integral

def trapezoid_inner_product_score(u, v, p=2):
  r"""
  Computes and inner product based score. This
  approximates the integral using a repeated trapezoid rule.
  Args:
    u: [batch x heads x seq1 x channels x ...]
    v: [batch x heads x seq2 x channels x ...]
  """
  assert p == 2
  local_product = einops.einsum(
    u, v, 'b h s1 ..., b h s2 ... -> b h s1 s2 ...')
  local_inner_product = local_product.sum(dim=4)
  dims_to_sum = list(range(4, local_inner_product.ndim))
  if not dims_to_sum:
    return local_inner_product
  
  integral = local_inner_product
  for i in range(len(dims_to_sum)):
    integral = torch.trapezoid(integral, dx=1/integral.size(-1))
  return integral

def self_attention(query, 
                   key, 
                   value, 
                   bias: Optional[torch.Tensor] = None,
                   shifted = False,
                   score_method='reimann'):
  assert score_method in SCORE_METHODS
  if score_method == 'reimann':
    score = reimann_inner_product_score(query, key)
  elif score_method == 'trapezoid':
    score = trapezoid_inner_product_score(query, key)
  score = score / math.sqrt(query.size(3))
  if bias is not None:
    if not shifted:
      score = score + bias
    else:
      # batch*nWindows x nHeads x nSeq1 x nSeq2 -> batch x nWindows x nHeads x nSeq1 x nSeq2
      batch_size = score.size(0)
      num_windows = bias.size(0)
      score = score.unflatten(0, (batch_size//num_windows, num_windows)) + bias
      score = score.flatten(0, 1)
  probabilities = softmax(score, dim=-1)
  return einops.einsum(
    probabilities, value, 'b h s1 s2, b h s2 ... -> b h s1 ...')