import pytest

import einops
import torch
from torch.nn.functional import interpolate, grid_sample

from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.grid.utility import cell_centered_unit_grid
#from mondrian.layers.linear_operator import (
#    SeparableInterpLinearOperator2d,
#    SplineInterpLinearOperator2d
#)

"""
def smooth_data(res):
    delta = 1 / res
    x = delta * (torch.arange(0, res) + 0.5)
    y = delta * (torch.arange(0, res) + 0.5)
    x, y = torch.meshgrid(x, y, indexing='xy')
    return einops.rearrange(torch.sin(x * y), 'h w -> () () () h w')

def interp(v, height, width):
    coords = 2 * cell_centered_unit_grid((height, width), device=v.device).permute(1, 2, 0) - 1
    coords = einops.repeat(coords, 'h w d -> b h w d', b=1)
    # The documentation explaining this is not super clear, but align_corners=True
    # should essentially makes it so the kernel's boundary valeus actually
    # correspond to [-1, 1] indices. This makes it so higher resolutions
    # do not extrapolate kernel values.
    return torch.nn.functional.grid_sample(v,
                                           coords, 
                                           align_corners=False, 
                                           mode='bicubic')

@pytest.mark.parametrize("data_func", [smooth_data])
def test_no(data_func):
    set_default_quadrature_method('simpson_13')
    #l = SeparableInterpLinearOperator2d(1, 1, bias=False, mode='bilinear')
    l = SplineInterpLinearOperator2d(1, 1, bias=False)
    
    coarse = l(data_func(64)).squeeze(1)
    fine = l(data_func(512)).squeeze(1)
    coarse_interp = interpolate(
        coarse, size=(fine.size(-2), fine.size(-1)), mode='bilinear'
    )
    
    print(fine.mean(), coarse_interp.mean())
    print(fine.min(), coarse_interp.min())
    print(fine.max(), coarse_interp.max())
    
    assert ((fine - coarse_interp) ** 2).sum() < 1e-9
"""