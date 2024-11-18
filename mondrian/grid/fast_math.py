r"""
Contains common functions that may warrant 'fast' implementations.
These typically take functions as input, rather than vectors.
"""

from mondrian.grid.attention._naive import (
  self_attention as naive_self_attention,
  reimann_inner_product_score as naive_inner_product_score
)

def inner_product_score(u, v, p=2):
  return naive_inner_product_score(u, v, p)

def is_power_of_2(x: int):
  assert isinstance(x, int)
  # gross way of checking number of 1 bits...,
  return bin(x).count('1') == 1

def attention(query, key, value, bias=None):
  return naive_self_attention(query, key, value, bias)