import pytest

import torch

from mondrian.models.vit_operator import ViTOperator2d

from util import available_devices

def test_vit_operator_init():
  vo = ViTOperator2d(in_channels=32,
                     out_channels=32, 
                     embed_dim=16, 
                     num_heads=4, 
                     num_layers=4, 
                     subdomain_size=1)
  assert vo.subdomain_size == (1, 1)
  
def test_vit_operator_forward_device():
  vo = ViTOperator2d(in_channels=32,
                     out_channels=32, 
                     embed_dim=16, 
                     num_heads=4, 
                     num_layers=4, 
                     subdomain_size=1).to('meta')
  v = torch.ones(4, 32, 32, 32).to('meta')
  u = vo(v, 2, 2)
  assert u.size() == v.size()

@pytest.mark.parametrize('device', available_devices())
def test_vit_operator_forward(device):
  vo = ViTOperator2d(in_channels=32,
                     out_channels=32, 
                     embed_dim=16, 
                     num_heads=4, 
                     num_layers=4, 
                     subdomain_size=1).to(device)
  v = torch.ones(4, 32, 32, 32).to(device)
  u = vo(v, 2, 2)
  assert u.size() == v.size()
  assert torch.all(u.isfinite())