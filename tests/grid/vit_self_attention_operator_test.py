import torch

from mondrian.grid.func_self_attention import FuncSelfAttention

def test_vit_sa_init():
  vit = FuncSelfAttention(embed_dim=32, num_heads=4, use_bias=False)
  assert vit.embed_dim == 32
  assert vit.num_heads == 4
  assert vit.use_bias == False
  
  vit = FuncSelfAttention(embed_dim=32, num_heads=4, use_bias=True)
  assert vit.embed_dim == 32
  assert vit.num_heads == 4
  assert vit.use_bias == True
  
def test_vit_sa_forward():
  vit = FuncSelfAttention(32, 4, use_bias=False)
  v = torch.ones(8, 4, 32, 16, 16)
  u = vit(v, 2, 2)
  assert u.size(0) == 8
  assert u.size(1) == 4
  assert u.size(2) == 32
  assert u.size(3) == 16
  assert u.size(4) == 16
  assert not u.isnan().any()
  
def test_vit_sa_forward():
  vit = FuncSelfAttention(32, 4, use_bias=True)
  v = torch.ones(8, 4, 32, 16, 16)
  u = vit(v, 2, 2)
  assert u.size(0) == 8
  assert u.size(1) == 4
  assert u.size(2) == 32
  assert u.size(3) == 16
  assert u.size(4) == 16
  assert not u.isnan().any()