import mondrian.grid.fast_math as fm

import torch

def test_inner_product_score_1d():
  u = torch.zeros(8, 4, 64, 32, 64)
  v = torch.ones(8, 4, 64, 32, 64)
  
  inner = fm.inner_product_score(u, v)
  assert inner.sum() == 0
  assert inner.size(0) == 8
  assert inner.size(1) == 4
  assert inner.size(2) == 64
  assert inner.size(3) == 64
  assert inner.ndim == 4
  
def test_inner_product_score_2d():
  u = torch.zeros(8, 4, 64, 32, 16, 16)
  v = torch.ones(8, 4, 64, 32, 16, 16)
  
  inner = fm.inner_product_score(u, v)
  assert inner.sum() == 0
  assert inner.size(0) == 8
  assert inner.size(1) == 4
  assert inner.size(2) == 64
  assert inner.size(3) == 64
  assert inner.ndim == 4
  
def test_inner_product_score_2d():
  u = torch.ones(8, 4, 64, 32, 16, 16)
  v = torch.ones(8, 4, 64, 32, 16, 16)
  
  inner = fm.inner_product_score(u, v)
  # The output is 64 x 64 x 16 x 16 since reduces over channels (32)
  # and then computes for all pairs (64 x 64).
  assert inner[0, 0].sum() == 2
  assert inner.size(0) == 8
  assert inner.size(1) == 4
  assert inner.size(2) == 64
  assert inner.size(3) == 64
  assert inner.ndim == 4
  
def test_self_attention():
  q = torch.zeros(8, 4, 8, 16, 32, 32)
  k = torch.zeros(8, 4, 8, 16, 32, 32)
  v = torch.zeros(8, 4, 8, 16, 32, 32)
  
  sa = fm.self_attention(q, k, v)
  assert sa.sum() == 0