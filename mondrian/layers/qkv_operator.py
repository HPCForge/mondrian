import logging
from functools import partial

from .linear_operator import (
    LinearOperator2d,
    SeparableLinearOperator2d,
    LowRankLinearOperator2d
)

from .spectral_conv import SimpleSpectralConv2d

LINEAR_OPERATOR = 'linear_operator'
SEPARABLE_LINEAR_OPERATOR = 'separable_operator'
LOW_RANK_LINEAR_OPERATOR = 'low_rank_linear_operator'
SPECTRAL_CONV = 'spectral_conv'
QKV_OPERATORS = [LINEAR_OPERATOR, SEPARABLE_LINEAR_OPERATOR, LOW_RANK_LINEAR_OPERATOR, SPECTRAL_CONV]

_DEFAULT_QKV_OPERATOR = LINEAR_OPERATOR

def set_default_qkv_operator(operator: str):
    global _DEFAULT_QKV_OPERATOR
    assert operator is None or operator in QKV_OPERATORS
    logging.info(f'changing _DEFAULT_LINEAR_OPERATOR to {operator}')
    _DEFAULT_QKV_OPERATOR = operator
    
def get_default_qkv_operator(*args, **kwargs):
    lookup = {
        LINEAR_OPERATOR: LinearOperator2d,
        SEPARABLE_LINEAR_OPERATOR: SeparableLinearOperator2d,
        LOW_RANK_LINEAR_OPERATOR: LowRankLinearOperator2d,
        SPECTRAL_CONV: SimpleSpectralConv2d
    }
    assert _DEFAULT_QKV_OPERATOR in lookup.keys()
    return lookup[_DEFAULT_QKV_OPERATOR](*args, **kwargs)