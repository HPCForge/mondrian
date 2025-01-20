from neuralop.models import FNO
from .factformer.factformer import FactorizedTransformer, FactFormer2D
from .ffno.ffno import FNOFactorized2DBlock
from .transformer.vit_operator_2d import ViTOperator2d, ViTOperatorFixedPosEmbedding2d
from .transformer.galerkin_transformer_2d import GalerkinTransformer2d
from .transformer.point_transformer_2d import PointTransformer2d
from .transformer.swin_sa_operator_2d import SwinSAOperator2d
from .transformer.unet_operator_2d import UnetOperator2d

_VIT_OPERATOR_2D = "vit_operator_2d"
_SWIN_SA_OPERATOR_2D = "swin_sa_operator_2d"
_UNET_OPERATOR_2D = 'unet_operator_2d'
_VIT_OPERATOR_FIXED_POS_2D = 'vit_operator_fixed_pos_2d'
_GALERKIN_TRANSFORMER_2D = "galerkin_transformer_2d"
_POINT_TRANSFORMER_2D = "point_transformer_2d"
_FACTFORMER_2D = 'factformer_2d'

_FNO = "fno"
_FFNO = "ffno"

MODELS = [
    _VIT_OPERATOR_2D, 
    _SWIN_SA_OPERATOR_2D,
    _UNET_OPERATOR_2D,
    _VIT_OPERATOR_FIXED_POS_2D,
    _GALERKIN_TRANSFORMER_2D, 
    _POINT_TRANSFORMER_2D, 
    _FACTFORMER_2D, 
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
            channel_heads=model_cfg.channel_heads,
            x_heads=model_cfg.x_heads,
            y_heads=model_cfg.y_heads,
            num_layers=model_cfg.num_layers,
            max_seq_len=model_cfg.max_seq_len,
            subdomain_size=model_cfg.subdomain_size,
        )
    if model_cfg.name == _SWIN_SA_OPERATOR_2D:
        return SwinSAOperator2d(
            in_channels,
            out_channels,
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            head_split=model_cfg.head_split,
            num_layers=model_cfg.num_layers,
            window_size=model_cfg.window_size,
            shift_size=model_cfg.shift_size,
            n_sub_x=model_cfg.n_sub_x,
            n_sub_y=model_cfg.n_sub_y,
            subdomain_size=model_cfg.subdomain_size,
        )
    if model_cfg.name == _UNET_OPERATOR_2D:
        return UnetOperator2d(
            in_channels,
            out_channels,
            embed_dim=model_cfg.embed_dim,
            channel_heads=model_cfg.channel_heads,
            x_heads=model_cfg.x_heads,
            y_heads=model_cfg.y_heads,
            num_layers=model_cfg.num_layers,
            max_seq_len=model_cfg.max_seq_len,
            subdomain_size=model_cfg.subdomain_size,
        )
    if model_cfg.name == _VIT_OPERATOR_FIXED_POS_2D:
        return ViTOperatorFixedPosEmbedding2d(
            in_channels,
            out_channels,
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            head_split=model_cfg.head_split,
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
    if model_cfg.name == _POINT_TRANSFORMER_2D:
        return PointTransformer2d(
            in_channels,
            out_channels,
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
        )
    if model_cfg.name == _FACTFORMER_2D:
        return FactFormer2D(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=model_cfg.dim,
            depth=model_cfg.depth,
            dim_head=model_cfg.dim_head,
            heads=model_cfg.heads,
            pos_in_dim=2,
            pos_out_dim=2,
            kernel_multiplier=2,
            positional_embedding='rotary'
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
