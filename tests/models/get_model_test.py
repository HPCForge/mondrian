from omegaconf import OmegaConf
from mondrian.models import get_model


def test_fno():
    model_cfg = OmegaConf.create(
        """
    name: 'fno'
    n_modes: [16, 16]
    hidden_channels: 64
    num_layers: 4
    domain_size: null
    norm: 'instance_norm'
    domain_padding: 0.25
  """
    )
    fno = get_model(3, 5, model_cfg)
    assert fno.in_channels == 3


def test_ffno():
    model_cfg = OmegaConf.create(
        """
    name: 'ffno'
    n_modes: 16
    hidden_channels: 64
    num_layers: 4
    layer_norm: True
    domain_padding: 0.25
  """
    )
    ffno = get_model(3, 5, model_cfg)
    assert ffno.in_channels == 3
    assert ffno.modes == 16


def test_vit_operator():
    model_cfg = OmegaConf.create(
        """
    name: 'vit_operator_2d'
    embed_dim: 32
    num_heads: 4
    head_split: 'channel'
    score_method: 'reimann'
    num_layers: 4
    max_seq_len: 32
    subdomain_size: [1, 1]
    domain_size: [4, 8]
  """
    )
    vit = get_model(3, 5, model_cfg)
    assert vit.in_channels == 3
    assert vit.sub_size_x == 1


def test_galerkin_transformer():
    model_cfg = OmegaConf.create(
        """
    name: 'galerkin_transformer_2d'
    embed_dim: 32
    num_heads: 4
    num_layers: 4
    quadrature_method: 'reimann'
  """
    )
    vit = get_model(3, 5, model_cfg)
    assert vit.in_channels == 3
