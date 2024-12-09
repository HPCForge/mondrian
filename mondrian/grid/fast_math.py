r"""
Contains common functions that may warrant 'fast' implementations.
These typically take functions as input, rather than vectors.
"""

from typing import List, Tuple

import torch

from mondrian.grid.attention._naive import (
  self_attention as naive_self_attention,
  reimann_inner_product_score as naive_inner_product_score
)

def inner_product_score(u, v, p=2):
  return naive_inner_product_score(u, v, p)

def attention(query, key, value, bias=None, score_method='reimann'):
  return naive_self_attention(query, key, value, bias, score_method)