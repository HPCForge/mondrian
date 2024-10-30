import torch

from mondrian.grid.vit_self_attention_operator import ViTSelfAttentionOperator

def test_vit_sa_init():
  vit = ViTSelfAttentionOperator(embed_dim=32, num_heads=4)
  assert vit.embed_dim == 32
  assert vit.num_heads == 4
  
def test_vit_sa_forward():
  vit = ViTSelfAttentionOperator(32, 4)
  v = torch.ones(8, 4, 32, 16, 16)
  u = vit(v, 2, 2)
  assert u.size(0) == 8
  assert u.size(1) == 4
  assert u.size(2) == 32
  assert u.size(3) == 16
  assert u.size(4) == 16
  assert not u.isnan().any()