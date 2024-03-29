from neuralop.models import FNO
from mondrian_lib.fdm.models.pdebench.unet2d import UNet2d 
from mondrian_lib.fdm.models.ffno.ffno import FNOFactorized2DBlock

_UNET_BENCH = 'pdebench_unet'

_FNO = 'fno'
_FFNO = 'ffno'

_MODEL_LIST = [
    _UNET_BENCH,

    _FNO,
    _FFNO
]

def get_model(in_channels, out_channels, model_cfg, device):
    model_name = model_cfg.model_name
    assert model_name in _MODEL_LIST
    if model_name == _UNET_BENCH:
        model = UNet2d(in_channels=in_channels,
                       out_channels=out_channels,
                       init_features=model_cfg.init_features)
    elif model_name == _FNO:
        model = FNO(n_modes=model_cfg.n_modes,
                    hidden_channels=model_cfg.hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels)
    elif model_name == _FFNO:
        model = FNOFactorized2DBlock(in_channels=in_channels,
                                     out_channels=out_channels,
                                     # FNO internally halves the number of modes,
                                     # but FFNO does not so we do it manually.
                                     modes=model_cfg.modes // 2,
                                     width=model_cfg.width,
                                     layer_norm=model_cfg.layer_norm)

    return model.to(device).float()
