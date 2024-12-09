import torch

from mondrian.grid.attention._naive import (
  reimann_inner_product_score,
  trapezoid_inner_product_score
)

Y, X = 32, 32
size = (4, 4, 32, 16, Y, X)

def test_compare_reimann_trapezoid():
  u = torch.ones(size)
  v = torch.ones(size)
  r = reimann_inner_product_score(u, v)
  t = trapezoid_inner_product_score(u, v)
  assert torch.allclose(r, t)
  
  x, y = torch.meshgrid(torch.linspace(0, 2 * torch.pi, X), torch.linspace(0, 2 * torch.pi, Y), indexing='xy')
  u = torch.sin(x + y).repeat(4, 4, 32, 16, 1, 1)
  v = torch.ones(size)
  r = reimann_inner_product_score(u, v)
  t = trapezoid_inner_product_score(u, v)
  assert torch.allclose(r, t, atol=1e-3)
  # These should both be close to zero...
  # but trapezoid should be closer on a sufficiently fine grid...
  assert torch.all(abs(t) < abs(r))
  
  # randn would look like a wild function,
  # so these should NOT be close!
  u = torch.randn(size)
  v = torch.randn(size)
  r = reimann_inner_product_score(u, v)
  t = trapezoid_inner_product_score(u, v)
  assert not torch.allclose(r, t)