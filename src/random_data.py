import numpy
import scipy.fftpack
from numpy.random import default_rng
import torch
import matplotlib.pyplot as plt

from mondrian.attention.func_self_attention import FuncSelfAttention
from mondrian.layers.linear_operator import LinearOperator2d, LowRankLinearOperator2d
from mondrian.models.transformer.vit_operator_2d import ViTOperator2d
from mondrian.grid.quadrature import set_default_quadrature_method, simpsons_13_quadrature_weights, get_unit_quadrature_weights
from mondrian.layers.qkv_operator import set_default_qkv_operator, get_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator, get_default_feed_forward_operator

def kat_data(d):
    x = torch.linspace(0, 1, d)
    return x ** 3 * torch.sqrt(x)

def sin_data(d):
    delta_x = torch.pi / d
    delta_y = 1 / d
    x = delta_x * (torch.arange(0, d) + 0.5)
    y = delta_y * (torch.arange(0, d) + 0.5)
    x, y = torch.meshgrid(x, y, indexing='xy')
    return torch.exp(x * y) * torch.cos(x * y).unsqueeze(0).unsqueeze(0)

def integrate(x):
    qw = simpsons_13_quadrature_weights((1, 1), (x.size(-2), x.size(-1)), device=x.device)
    return (x * qw).sum()

def test(model, data_func, default_quad):
    r"""
    This basically checks how quickly a model spits out a
    decent approximation of the high resolution data, when using different integration methods.
    I.e., a simpsons method is accurate quickly, so it should reach a better approximation, at lower resolutions.
    """
    set_default_quadrature_method(default_quad)
    target = model(data_func(400).cuda())
    from mondrian.grid.quadrature import DEFAULT_QUADRATURE_METHOD
    print(default_quad, DEFAULT_QUADRATURE_METHOD)
    print(f'  - target: {target.mean()}')
    for res in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 51, 71, 91, 129]:
        data = data_func(res).cuda()
        out = model(data)
        print(f' - {res} estimate: {integrate(out)}, {integrate(target)} error: {abs(integrate(target) - integrate(out))}')

def main():
    set_default_qkv_operator('low_rank_linear_operator')
    set_default_feed_forward_operator('low_rank_neural_operator')
    torch.manual_seed(0)
    
    data_func = sin_data

    with torch.no_grad():    
        model = get_default_qkv_operator(1, 1, bias=True).cuda()
        test(model, data_func, 'reimann')
        test(model, data_func, 'trapezoid')
        test(model, data_func, 'simpson_13')
        
    sample = sin_data(128)
    
    plt.imshow(sample.squeeze().detach().cpu())
    plt.savefig('sample.png')
    
if __name__ == '__main__':
    main()

