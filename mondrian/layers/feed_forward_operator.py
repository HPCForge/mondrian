import logging

from .linear_operator import (
    NeuralOperator,
    RandomProjectNeuralOperator,
    SeparableRandomProjectNeuralOperator,
    LowRankNeuralOperator,
    LowRankInterpNeuralOperator,
    SplineInterpNeuralOperator,
    AttentionNeuralOperator
)

from .spectral_conv import SpectralConvNeuralOperator

NEURAL_OPERATOR = 'neural_operator'
RANDOM_PROJECT_NEURAL_OPERATOR = 'random_project_neural_operator'
SEPARABLE_RANDOM_PROJECT_NEURAL_OPERATOR = 'separable_random_project_neural_operator'
LOW_RANK_NEURAL_OPERATOR = 'low_rank_neural_operator'
LOW_RANK_INTERP_NEURAL_OPERATOR = 'low_rank_interp_neural_operator'
SPLINE_INTERP_NEURAL_OPERATOR = 'spline_interp_neural_operator'
ATTENTION_NEURAL_OPERATOR = 'attention_neural_operator'
SPECTRAL_CONV_NEURAL_OPERATOR = 'spectral_conv_neural_operator'
FEED_FORWARD_OPERATORS = [
    NEURAL_OPERATOR,
    RANDOM_PROJECT_NEURAL_OPERATOR,
    SEPARABLE_RANDOM_PROJECT_NEURAL_OPERATOR,
LOW_RANK_NEURAL_OPERATOR,
    LOW_RANK_INTERP_NEURAL_OPERATOR,
    SPLINE_INTERP_NEURAL_OPERATOR,
    ATTENTION_NEURAL_OPERATOR,
    SPECTRAL_CONV_NEURAL_OPERATOR
]

_DEFAULT_FEED_FORWARD_OPERATOR = NEURAL_OPERATOR

_lookup = {
    NEURAL_OPERATOR: NeuralOperator,
    RANDOM_PROJECT_NEURAL_OPERATOR: RandomProjectNeuralOperator,
    SEPARABLE_RANDOM_PROJECT_NEURAL_OPERATOR: SeparableRandomProjectNeuralOperator,
    LOW_RANK_NEURAL_OPERATOR: LowRankNeuralOperator,
    LOW_RANK_INTERP_NEURAL_OPERATOR: LowRankInterpNeuralOperator,
    SPLINE_INTERP_NEURAL_OPERATOR: SplineInterpNeuralOperator,
    ATTENTION_NEURAL_OPERATOR: AttentionNeuralOperator,
    SPECTRAL_CONV_NEURAL_OPERATOR: SpectralConvNeuralOperator
}

def set_default_feed_forward_operator(operator: str):
    global _DEFAULT_FEED_FORWARD_OPERATOR
    assert operator is None or operator in FEED_FORWARD_OPERATORS
    logging.info(f'changing _DEFAULT_NEURAL_OPERATOR to {operator}')
    _DEFAULT_FEED_FORWARD_OPERATOR = operator
    
def get_feed_forward_operator(name, *args, **kwargs):
    print(name)
    assert name in _lookup.keys()
    return _lookup[name](*args, **kwargs)
    
def get_default_feed_forward_operator(*args, **kwargs):

    assert _DEFAULT_FEED_FORWARD_OPERATOR in _lookup.keys()
    return _lookup[_DEFAULT_FEED_FORWARD_OPERATOR](*args, **kwargs)