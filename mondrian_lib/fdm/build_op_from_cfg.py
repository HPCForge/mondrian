from mondrian_lib.fdm.dd_op import DDOpAdditive
from mondrian_lib.fdm.kernel import (
    LowRankKernel,
    LowRankLinearKernel,
    NystromNonLinearKernel,
    NystromLinearKernel
)

_DD_OP_ADDITIVE = 'additive'

_OP_TYPES = [
    _DD_OP_ADDITIVE
]

_LOW_RANK_KERNEL = 'low_rank'
_LOW_RANK_LINEAR_KERNEL = 'low_rank_linear'

_KERNEL_TYPES = [
    _LOW_RANK_KERNEL,
    _LOW_RANK_LINEAR_KERNEL
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

    if op_cfg.kernel_name == _LOW_RANK_KERNEL:
        kernel = LowRankKernel(in_channels,
                               out_channels,
                               hidden_channels,
                               rank=op_cfg.rank)
    elif op_cfg.kernel_name == _LOW_RANK_LINEAR_KERNEL:
        kernel = LowRankLinearKernel(in_channels,
                                     out_channels,
                                     hidden_channels,
                                     rank=op_cfg.rank)

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
