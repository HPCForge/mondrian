import logging

from .linear_operator import (
    NeuralOperator,
    SeparableNeuralOperator,
    LowRankNeuralOperator,
)

from .spectral_conv import SpectralConvNeuralOperator

NEURAL_OPERATOR = 'neural_operator'
SEPARABLE_NEURAL_OPERATOR = 'separable_neural_operator'
LOW_RANK_NEURAL_OPERATOR = 'low_rank_neural_operator'
SPECTRAL_CONV_NEURAL_OPERATOR = 'spectral_conv_neural_operator'
FEED_FORWARD_OPERATORS = [NEURAL_OPERATOR, SEPARABLE_NEURAL_OPERATOR, LOW_RANK_NEURAL_OPERATOR, SPECTRAL_CONV_NEURAL_OPERATOR]

_DEFAULT_FEED_FORWARD_OPERATOR = NEURAL_OPERATOR

def set_default_feed_forward_operator(operator: str):
    global _DEFAULT_FEED_FORWARD_OPERATOR
    assert operator is None or operator in FEED_FORWARD_OPERATORS
    logging.info(f'changing _DEFAULT_NEURAL_OPERATOR to {operator}')
    _DEFAULT_FEED_FORWARD_OPERATOR = operator
    
def get_default_feed_forward_operator(*args, **kwargs):
    lookup = {
        NEURAL_OPERATOR: NeuralOperator,
        SEPARABLE_NEURAL_OPERATOR: SeparableNeuralOperator,
        LOW_RANK_NEURAL_OPERATOR: LowRankNeuralOperator,
        SPECTRAL_CONV_NEURAL_OPERATOR: SpectralConvNeuralOperator
    }
    assert _DEFAULT_FEED_FORWARD_OPERATOR in lookup.keys()
    return lookup[_DEFAULT_FEED_FORWARD_OPERATOR](*args, **kwargs)