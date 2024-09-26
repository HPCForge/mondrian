from mondrian_lib.fdm.dd_op import DDOpAdditive
from mondrian_lib.fdm.kernel import (
    FullRankLinearKernel,
    LowRankKernel,
    LowRankLinearKernel,
    NystromNonLinearKernel,
    NystromLinearKernel
)

_DD_OP_ADDITIVE = 'additive'

_OP_TYPES = [
    _DD_OP_ADDITIVE
]

_FULL_RANK_LINEAR_KERNEL = 'full_rank_linear'
_LOW_RANK_KERNEL = 'low_rank'
_LOW_RANK_LINEAR_KERNEL = 'low_rank_linear'
_NYSTROM_KERNEL = 'nystrom'

_SPECTRAL_CONV = 'spectral_conv'
_FACTORIZED_SPECTRAL_CONV = 'factorized_spectral_conv'

_KERNEL_TYPES = [
    _FULL_RANK_LINEAR_KERNEL,
    _LOW_RANK_KERNEL,
    _LOW_RANK_LINEAR_KERNEL,
    _NYSTROM_KERNEL,
    _SPECTRAL_CONV,
    _FACTORIZED_SPECTRAL_CONV
]

def build_op_from_cfg(op_cfg,
                      in_channels,
                      out_channels,
                      hidden_channels,
                      domain_size_x,
                      domain_size_y,
                      use_coarse_op):
    r"""
    Construct a DDOp with some Kernel based on the configuration
    """
    assert op_cfg.op_name in _OP_TYPES
    assert op_cfg.kernel_name in _KERNEL_TYPES

    if op_cfg.kernel_name == _FULL_RANK_LINEAR_KERNEL:
        kernel = FullRankLinearKernel(in_channels,
                                      out_channels,
                                      num_filters=op_cfg.num_filters)
    elif op_cfg.kernel_name == _LOW_RANK_KERNEL:
        kernel = LowRankKernel(in_channels + 2,
                               out_channels,
                               hidden_channels,
                               rank=op_cfg.rank)
    elif op_cfg.kernel_name == _LOW_RANK_LINEAR_KERNEL:
        kernel = LowRankLinearKernel(in_channels,
                                     out_channels,
                                     hidden_channels,
                                     rank=op_cfg.rank)
    elif op_cfg.kernel_name == _NYSTROM_KERNEL:
        kernel = NystromNonLinearKernel(in_channels,
                                        out_channels,
                                        hidden_channels,
                                        sample_size=op_cfg.sample_size)
    elif op_cfg.kernel_name == _SPECTRAL_CONV:
        from neuralop.layers.spectral_convolution import SpectralConv
        kernel = SpectralConv(in_channels,
                                out_channels,
                                n_modes=op_cfg.n_modes)
    elif op_cfg.kernel_name == _FACTORIZED_SPECTRAL_CONV:
        from mondrian_lib.fdm.models.ffno.ffno import SpectralConv2d
        kernel = SpectralConv2d(in_channels,
                                out_channels,
                                n_modes=op_cfg.n_modes // 2,
                                permute=True)

    if op_cfg.op_name == _DD_OP_ADDITIVE:
        op = DDOpAdditive(kernel,
                          hidden_channels,
                          domain_size_x,
                          domain_size_y,
                          op_cfg.subdomain_size_x,
                          op_cfg.subdomain_size_y,
                          op_cfg.overlap_x,
                          op_cfg.overlap_y,
                          use_coarse_op,
                          op_cfg.use_padding)

    return op
