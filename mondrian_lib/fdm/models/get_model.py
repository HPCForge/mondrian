from neuralop.models import FNO
from mondrian_lib.fdm.models.pdebench.unet2d import UNet2d 
from mondrian_lib.fdm.models.ffno.ffno import FNOFactorized2DBlock
from mondrian_lib.fdm.models.ddno.ddno import DDNO
from mondrian_lib.fdm.models.ddno.ddno_ref import DDNORef

_UNET_BENCH = 'pdebench_unet'

_FNO = 'fno'
_FFNO = 'ffno'

_DDNO = 'ddno'
_DDNO_REF = 'ddno_ref'

_MODEL_LIST = [
    _UNET_BENCH,

    _FNO,
    _FFNO,

    _DDNO,
    _DDNO_REF
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
    elif model_name == _DDNO:
        model = DDNO(in_channels=in_channels,
                     out_channels=out_channels,
                     hidden_channels=model_cfg.hidden_channels,
                     layers=model_cfg.layers,
                     domain_size_x=1,
                     domain_size_y=1,
                     op_cfg=model_cfg.op_cfg)
    elif model_name == _DDNO_REF:
        model = DDNORef(in_channels=in_channels,
                     out_channels=out_channels,
                     hidden_channels=model_cfg.hidden_channels,
                     layers=model_cfg.layers,
                     n_subdomain_x=4,
                     n_subdomain_y=4)

    return model.to(device).float()
