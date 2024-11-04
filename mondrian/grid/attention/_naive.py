import math
import einops
import torch
from torch.nn.functional import softmax

def inner_product_score(u, v, p=2):
  r"""
  Args:
    u: [batch x heads x seq x channels x ...]
  """
  assert u.size() == v.size()
  return _naive_inner_product_score(u, v, p)
  
def _naive_inner_product_score(u, v, p=2):
  r"""
  Computes and inner product based score. This
  approximates the integral with a naive sum.
  Args:
    u: [batch x heads x seq x channels x ...]
    v: [batch x heads x seq x channels x ...]
  """
  assert p == 2
  local_product = einops.einsum(
    u, v, 'b h s1 c ..., b h s2 c ... -> b h s1 s2 c ...')
  local_inner_product = local_product.sum(dim=4)
  numel = local_inner_product[0,0,0,0].numel()
  assert numel > 0
  dims_to_sum = list(range(4, local_inner_product.ndim))
  integral = (local_inner_product / numel).sum(dim=dims_to_sum)
  return integral

def self_attention(query, key, value, bias=None):
  score = inner_product_score(query, key)
  if bias is not None:
    score = score + bias
  probabilities = softmax(score, dim=-1)
  return einops.einsum(
    probabilities, value, 'b h s1 s2, b h s2 ... -> b h s1 ...')