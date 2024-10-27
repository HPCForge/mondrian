import torch

from mondrian.models.vit_operator import ViTOperator

def test_vit_operator_init():
  vo = ViTOperator(embed_dim=32, num_heads=4, num_layers=4, subdomain_size=1)
  assert vo.subdomain_size == 1
  
def test_vit_operator_forward():
  vo = ViTOperator(embed_dim=32, num_heads=4, num_layers=1, subdomain_size=1)
  v = torch.ones(4, 32, 32, 32)
  u = vo(v, 2, 2)
  assert u.size() == v.size()
  assert torch.all(u.isfinite())