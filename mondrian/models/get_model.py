from neuralop.models import FNO

from .vit_operator_2d import ViTOperator2d
from .galerkin_transformer_2d import GalerkinTransformer2d
from .ffno.ffno import FNOFactorized2DBlock

_VIT_OPERATOR_2D = "vit_operator_2d"
_GALERKIN_TRANSFORMER_2D = "galerkin_transformer_2d"

_FNO = "fno"
_FFNO = "ffno"

MODELS = [_VIT_OPERATOR_2D, _GALERKIN_TRANSFORMER_2D, _FNO, _FFNO]


def get_model(in_channels, out_channels, model_cfg):
    assert model_cfg.name in MODELS
    if model_cfg.name == _VIT_OPERATOR_2D:
        return ViTOperator2d(
            in_channels,
            out_channels,
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            head_split=model_cfg.head_split,
            score_method=model_cfg.score_method,
            num_layers=model_cfg.num_layers,
            max_seq_len=model_cfg.max_seq_len,
            subdomain_size=model_cfg.subdomain_size,
        )
    if model_cfg.name == _GALERKIN_TRANSFORMER_2D:
        return GalerkinTransformer2d(
            in_channels,
            out_channels,
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
        )
    if model_cfg.name == _FNO:
        return FNO(
            model_cfg.n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=model_cfg.hidden_channels,
            num_layers=model_cfg.num_layers,
            norm=model_cfg.norm,
            domain_padding=model_cfg.domain_padding,
        )
    if model_cfg.name == _FFNO:
        return FNOFactorized2DBlock(
            in_channels,
            out_channels,
            model_cfg.n_modes,
            hidden_channels=model_cfg.hidden_channels,
            n_layers=model_cfg.num_layers,
            domain_padding=model_cfg.domain_padding,
            layer_norm=model_cfg.layer_norm,
        )
