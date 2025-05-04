import einops
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

def sin_data(d):
    delta = 1 / d
    x = delta * (torch.arange(0, d) + 0.5)
    x = 1.5 * x #- 1.5
    x, y = torch.meshgrid(x, x, indexing='xy')
    return torch.exp(1.2 * x * y).unsqueeze(0).unsqueeze(0)

def poly_data(d):
    delta = 1 / d
    x = 2 * delta * (torch.arange(0, d) + 0.5)
    y = 4 * delta * (torch.arange(0, d) + 0.5) + 1
    x, y = torch.meshgrid(x, y, indexing='xy')
    return einops.rearrange(x ** 2 * y ** 3, 'h w -> () () h w')
    #return einops.rearrange(15 * x**2 + 17 * x * y + 24 * y**2 + x + y + 1, 'h w -> () () h w')

def run_model(model, data, steps):
    for _ in range(steps):
        data = model(data)
    return data

def test(model, data_func, default_quad):
    r"""
    This basically checks how quickly a model spits out a
    decent approximation of the high resolution data, when using different integration methods.
    I.e., a simpsons method is accurate quickly, so it should reach a better approximation, at lower resolutions.
    """
    set_default_quadrature_method('simpson_13')
    steps = 1
    target = run_model(model, data_func(64).cuda(), steps)
    
    set_default_quadrature_method(default_quad)
    from mondrian.grid.quadrature import DEFAULT_QUADRATURE_METHOD
    print(default_quad, DEFAULT_QUADRATURE_METHOD)
    print(f'  - target: {target.mean()}')
    for res in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32]:
        data = data_func(res).cuda()
        out = run_model(model, data, steps)
        print(f' - {res} estimate: {integrate(out)}, {integrate(target)} error: {abs(integrate(target) - integrate(out))}')

def main():
    set_default_qkv_operator('linear_operator')
    set_default_feed_forward_operator('neural_operator')
    torch.manual_seed(0)
    
    data_func = poly_data

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

