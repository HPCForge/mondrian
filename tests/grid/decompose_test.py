import torch 

from mondrian.grid.decompose import decompose2d, recompose2d

def test_decompose_2d_square():
  v = torch.zeros(2, 4, 32, 32)
  d = decompose2d(v, 4, 4)
  
  # batch and channels unchanged
  assert d.size(0) == 2
  assert d.size(2) == 4
  
  # assert 8 patches in vertical / horizontal direction
  # for sequence length of 64
  assert d.size(1) == 16
  assert d.size(3) == 8
  assert d.size(4) == 8
  
def test_decompose_2d_rectangle():
  v = torch.zeros(2, 4, 32, 32)
  d = decompose2d(v, 4, 8)
  
  # batch and channels unchanged
  assert d.size(0) == 2
  assert d.size(2) == 4
  
  # assert 8 patches in vertical / horizontal direction
  # for sequence length of 64
  assert d.size(1) == 32
  assert d.size(3) == 4
  assert d.size(4) == 8
  
def test_decompose_2d_reconstruct():
  v = torch.randn(2, 4, 32, 32)
  d = decompose2d(v, 2, 2)
  assert d.size(1) == 4
  assert d.size(3) == 16
  assert d.size(4) == 16
  
  # check that the decomposition preserves ordering.
  assert torch.allclose(v[:, :, 0:16, 0:16], d[:, 0, :, :, :])
  assert torch.allclose(v[:, :, 0:16, 16:32], d[:, 1, :, :, :])
  assert torch.allclose(v[:, :, 16:32, 0:16], d[:, 2, :, :, :])
  assert torch.allclose(v[:, :, 16:32, 16:32], d[:, 3, :, :, :])
  assert not torch.allclose(v[:, :, 0:16, 16:32], d[:, 3, :, :, :])
  
def test_recompose_2d_small():
  v = torch.randn(2, 4, 32, 32)
  d = decompose2d(v, 4, 4)
  r = recompose2d(d, 4, 4)
  
  assert r.size() == v.size()
  assert torch.allclose(v, r)
  
def test_recompose_2d_large_rectangle():
  v = torch.randn(8, 16, 256, 512)
  d = decompose2d(v, 4, 8)
  r = recompose2d(d, 4, 8)
  
  assert r.size() == v.size()
  assert torch.allclose(v, r)
  
def test_recompose_2d_large_rectangle():
  v = torch.randn(8, 16, 128, 1024)
  d = decompose2d(v, 4, 8)
  r = recompose2d(d, 4, 8)
  
  assert r.size() == v.size()
  assert torch.allclose(v, r)