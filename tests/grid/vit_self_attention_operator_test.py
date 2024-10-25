import torch

from mondrian.grid.vit_self_attention_operator import ViTSelfAttentionOperator

def test_vit_sa_init():
  vit = ViTSelfAttentionOperator(32, 4)
  assert vit.embed_dim == 32
  assert vit.num_heads == 4
  
def test_vit_sa_forward():
  vit = ViTSelfAttentionOperator(32, 4)
  v = torch.ones(8, 16, 32, 16, 16)
  u = vit(v)
  assert u.size(1) == 16
  assert u.size(2) == 32
  assert u.size(3) == 16
  assert u.size(4) == 16