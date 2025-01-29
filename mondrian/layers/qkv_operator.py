import logging

from .linear_operator import (
    LinearOperator2d,
    RandomProjectLinearOperator,
    SeparableRandomProjectLinearOperator,
    LowRankLinearOperator2d,
    LowRankInterpLinearOperator2d,
    AttentionLinearOperator2d,
    SplineInterpLinearOperator2d
)

from .spectral_conv import SimpleSpectralConv2d

LINEAR_OPERATOR = 'linear_operator'
RANDOM_PROJECT_LINEAR_OPERATOR = 'random_project_linear_operator'
SEPARABLE_RANDOM_PROJECT_LINEAR_OPERATOR = 'separable_random_project_linear_operator'
SEPARABLE_LINEAR_OPERATOR = 'separable_operator'
LOW_RANK_LINEAR_OPERATOR = 'low_rank_linear_operator'
LOW_RANK_INTERP_LINEAR_OPERATOR = 'low_rank_interp_linear_operator'
SPLINE_INTERP_LINEAR_OPERATOR = 'spline_interp_linear_operator'
ATTENTION_LINEAR_OPERATOR = 'attention_linear_operator'
SPECTRAL_CONV = 'spectral_conv'
QKV_OPERATORS = [
    LINEAR_OPERATOR, 
    RANDOM_PROJECT_LINEAR_OPERATOR,
    SEPARABLE_RANDOM_PROJECT_LINEAR_OPERATOR,
    SEPARABLE_LINEAR_OPERATOR, 
    LOW_RANK_LINEAR_OPERATOR, 
    LOW_RANK_INTERP_LINEAR_OPERATOR,
    SPLINE_INTERP_LINEAR_OPERATOR,
    ATTENTION_LINEAR_OPERATOR,
    SPECTRAL_CONV
]

_DEFAULT_QKV_OPERATOR = LINEAR_OPERATOR

_lookup = {
        LINEAR_OPERATOR: LinearOperator2d,
        RANDOM_PROJECT_LINEAR_OPERATOR: RandomProjectLinearOperator,
        SEPARABLE_RANDOM_PROJECT_LINEAR_OPERATOR: SeparableRandomProjectLinearOperator,
        LOW_RANK_LINEAR_OPERATOR: LowRankLinearOperator2d,
        LOW_RANK_INTERP_LINEAR_OPERATOR: LowRankInterpLinearOperator2d,
        SPLINE_INTERP_LINEAR_OPERATOR: SplineInterpLinearOperator2d,
        ATTENTION_LINEAR_OPERATOR: AttentionLinearOperator2d,
        SPECTRAL_CONV: SimpleSpectralConv2d
}

def set_default_qkv_operator(operator: str):
    global _DEFAULT_QKV_OPERATOR
    assert operator is None or operator in QKV_OPERATORS
    logging.info(f'changing _DEFAULT_LINEAR_OPERATOR to {operator}')
    _DEFAULT_QKV_OPERATOR = operator
    
def get_qkv_operator(name, *args, **kwargs):
    assert name in _lookup.keys()
    return _lookup[name](*args, **kwargs)
    
def get_default_qkv_operator(*args, **kwargs):
    
    assert _DEFAULT_QKV_OPERATOR in _lookup.keys()
    return _lookup[_DEFAULT_QKV_OPERATOR](*args, **kwargs)