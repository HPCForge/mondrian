import math
import einops

from torch.nn.functional import softmax

def inner_product_score(u, v, p=1):
  r"""
  Args:
    u: [batch x heads x seq x channels x ...]
  """
  assert u.size() == v.size()
  return _naive_inner_product_score(u, v, p)
  
def _naive_inner_product_score(u, v, p=1):
  r"""
  Computes and inner product based score. This
  approximates the integral with a naive sum.
  Args:
    u: [batch x heads x seq x channels x ...]
    v: [batch x heads x seq x channels x ...]
  """
  assert p == 1
  local_inner_product = einops.einsum(
    u, v, 'b h s1 c ..., b h s2 c ... -> b h s1 s2 c ...')
  local_inner_product = local_inner_product.sum(dim=4)
  numel = local_inner_product[4:].numel()
  dims_to_sum = list(range(4, local_inner_product.ndim))
  integral = (local_inner_product / numel).sum(dim=dims_to_sum)
  return integral

def self_attention(query, key, value):
  score = inner_product_score(query, key)
  probabilities = softmax(score, dim=-1)
  print(score.size(), probabilities.size())
  return einops.einsum(
    probabilities, value, 'b h s1 s2, b h s2 ... -> b h s1 ...')