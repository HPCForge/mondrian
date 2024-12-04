from neuralop.models import FNO

from .vit_operator_2d import ViTOperator2d
from .ffno.ffno import FNOFactorized2DBlock

_VIT_OPERATOR_2D = 'vit_operator_2d'

_FNO = 'fno'
_FFNO = 'ffno'

MODELS = [
  _VIT_OPERATOR_2D,
  _FNO,
  _FFNO
]

def get_model(in_channels, out_channels, model_cfg):
  assert model_cfg.name in MODELS
  if model_cfg.name == _VIT_OPERATOR_2D:
    return ViTOperator2d(
      in_channels,
      out_channels,
      embed_dim=model_cfg.embed_dim,
      num_heads=model_cfg.num_heads,
      head_split=model_cfg.head_split,
      num_layers=model_cfg.num_layers,
      subdomain_size=model_cfg.subdomain_size)
  if model_cfg.name == _FNO:
    return FNO(
      model_cfg.n_modes,
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_channels=model_cfg.hidden_channels,
      num_layers=model_cfg.num_layers)
  if model_cfg.name == _FFNO:
    return FNOFactorized2DBlock(
      in_channels,
      out_channels,
      model_cfg.n_modes,
      hidden_channels=model_cfg.hidden_channels,
      n_layers=model_cfg.num_layers)