from neuralop.models import FNO

from .vit_operator_2d import ViTOperator2d
from .ffno.ffno import FNOFactorized2DBlock

def get_model(in_channels, out_channels, model_cfg):
  if model_cfg.name == 'vit_operator_2d':
    return ViTOperator2d(
      in_channels,
      out_channels,
      model_cfg.embed_dim,
      model_cfg.num_heads,
      model_cfg.num_layers,
      model_cfg.subdomain_size)
  if model_cfg.name == 'fno':
    return FNO(
      model_cfg.n_modes,
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_channels=model_cfg.hidden_channels,
      num_layers=model_cfg.num_layers)
  if model_cfg.name == 'ffno':
    return FNOFactorized2DBlock(
      in_channels,
      out_channels,
      model_cfg.n_modes,
      hidden_channels=model_cfg.hidden_channels,
      n_layers=model_cfg.num_layers)