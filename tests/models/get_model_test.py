from omegaconf import OmegaConf
from mondrian.models import get_model

def test_fno():
  model_cfg = OmegaConf.create("""
    name: 'fno'
    n_modes: [16, 16]
    hidden_channels: 64
    num_layers: 4
  """)
  fno = get_model(3, 5, model_cfg)
  assert fno.in_channels == 3
  
def test_ffno():
  model_cfg = OmegaConf.create("""
    name: 'ffno'
    n_modes: 16
    hidden_channels: 64
    num_layers: 4
  """)
  ffno = get_model(3, 5, model_cfg)
  assert ffno.in_channels == 3
  assert ffno.modes == 16
  
def test_vit_operator():
  model_cfg = OmegaConf.create("""
    name: 'vit_operator_2d'
    embed_dim: 32
    num_heads: 4
    num_layers: 4
    subdomain_size: [1, 1]
  """)
  vit = get_model(3, 5, model_cfg)
  assert vit.in_channels == 3
  assert vit.sub_size_x == 1