import math

import einops
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from rotary_embedding_torch import RotaryEmbedding

from torch_cubic_spline_grids import CubicBSplineGrid2d, CubicCatmullRomGrid2d

from .log_cpb import FixedCPB2d
from ..grid.utility import cell_centered_unit_grid
from .seq_op import seq_op

class NeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__(
            LinearOperator2d(in_channels, hidden_channels, bias=True),
            torch.nn.Dropout(p=0.1),
            nn.GELU(),
            LinearOperator2d(hidden_channels, out_channels, bias=True),
            torch.nn.Dropout(p=0.1)
        )

class RandomProjectNeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels, n):
        super().__init__(
            RandomProjectLinearOperator(in_channels, hidden_channels, n, bias=True),
            torch.nn.Dropout(p=0.1),
            nn.GELU(),
            RandomProjectLinearOperator(hidden_channels, out_channels, n, bias=True),
            torch.nn.Dropout(p=0.1),
        )

class SeparableRandomProjectNeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels, n):
        super().__init__(
            SeparableRandomProjectLinearOperator(in_channels, hidden_channels, n, bias=True),
            torch.nn.Dropout(p=0.1),
            nn.GELU(),
            SeparableRandomProjectLinearOperator(hidden_channels, out_channels, n, bias=True),
            torch.nn.Dropout(p=0.1),
        )

class LowRankNeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__(
            LowRankLinearOperator2d(in_channels, hidden_channels, bias=True),
            nn.GELU(),
            LowRankLinearOperator2d(hidden_channels, out_channels, bias=True),
        )
        
class LowRankInterpNeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__(
            LowRankInterpLinearOperator2d(in_channels, hidden_channels, bias=True),
            nn.GELU(),
            LowRankInterpLinearOperator2d(hidden_channels, out_channels, bias=True),
        )

class SplineInterpNeuralOperator(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__(
            SplineInterpLinearOperator2d(in_channels, hidden_channels, bias=True),
            nn.GELU(),
            SplineInterpLinearOperator2d(hidden_channels, out_channels, bias=True),
        )
          
class AttentionNeuralOperator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.sa = AttentionLinearOperator2d(in_channels, hidden_channels, bias=True)
        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.sa(x)
        x = seq_op(self.cnn, x)
        return x

class LinearOperator2dBase(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, 1, 1))
            stdv = 1 / math.sqrt(self.in_channels)
            with torch.no_grad():
                self.bias.uniform_(-stdv, stdv)
        else:
            self.bias = None

class LinearOperator2d(LinearOperator2dBase):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__(in_channels, out_channels, bias)
        self.kernel = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, in_channels * out_channels)
        )
        
    def forward(self, v):
        height = v.size(-2)
        width = v.size(-1)
        coords = 2 * cell_centered_unit_grid((height, width, height, width), device=v.device).permute(1, 2, 3, 4, 0) - 1
        #coords1 = einops.rearrange(coords, 'h w d -> () () h w d')
        #coords2 = einops.rearrange(coords, 'h w d -> h w () () d')       
        #rel_dist = coords1 - coords2

        kernel = self.kernel(coords).reshape(height, width, height, width, self.out_channels, self.in_channels)
        v = einops.einsum(kernel, v, 'h1 w1 h2 w2 o i, ... i h2 w2 -> ... o h1 w1') / (height * width)
        if self.bias is not None:
            v = v + self.bias
        return v
    
class RandomProjectLinearOperator(LinearOperator2dBase):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 n,
                 bias):
        super().__init__(in_channels, out_channels, bias)
        
        self.in_channels = in_channels
        self.n = n
                
        self.params = nn.Parameter(
            torch.randn(
                self.n,
                self.out_channels,
                self.in_channels,
            ) * 1 / math.sqrt(self.n * self.in_channels))
        
        self.coeff_mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, self.n)
        )
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        
    def forward(self, v):
        height = v.size(-2)
        width = v.size(-1)
        coords = 2 * cell_centered_unit_grid((height, width, height, width), device=v.device).permute(1, 2, 3, 4, 0) - 1
        coeffs = self.coeff_mlp(coords)
        t = einops.einsum(
            coeffs, 
            self.params, 
            v, 
            'h1 w1 h2 w2 n, n o i, ... i h1 w1 -> ... o h2 w2') / (height * width)
        v = t + seq_op(self.conv, v)
        if self.bias is not None:
            v = v + self.bias
        return v
    
class SeparableRandomProjectLinearOperator(LinearOperator2dBase):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 n,
                 bias,
                 scale=None):
        super().__init__(in_channels, out_channels, bias)
        
        self.in_channels = in_channels
        self.n = n
                
        self.params = nn.Parameter(
            torch.randn(
                self.n,
                self.in_channels,
            ) * 1 / math.sqrt(self.n * self.in_channels))
        
        self.coeff_mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, self.n)
        )

        self.down = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        
        # if the output discretization is scaled to a different resolution,
        # we cannot apply a skip connection. Otherwise, we do use a skip.
        if scale is not None:
            self.skip = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.skip = None
        
        # This is used to project to different coarsities        
        self.scale = scale
        
    def forward(self, v):
        assert self.scale is None or isinstance(self.scale, (int, float))
        height = v.size(-2)
        width = v.size(-1)
        
        # optionally, can project to a different resolution grid.
        if self.scale is None:
            target_height = height
            target_width = width
        if isinstance(self.scale, (int, float)):
            target_height = height * self.scale
            target_width = width * self.scale
            
             
        coords = 2 * cell_centered_unit_grid((target_height, target_width, height, width), device=v.device).permute(1, 2, 3, 4, 0) - 1
        coeffs = self.coeff_mlp(coords)
        t = einops.einsum(
            coeffs, 
            self.params, 
            v,
            'h2 w2 h1 w1 n, n i, ... i h1 w1 -> ... i h2 w2') / (height * width)
        v = seq_op(self.down, t)
        if self.skip is not None:
            v = v + seq_op(self.skip, v)
        if self.bias is not None:
            v = v + self.bias
        return v

class LowRankLinearOperator2d(LinearOperator2dBase):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__(in_channels, out_channels, bias)
        self.rank = 16
        self.kernel1 = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, out_channels * self.rank)
        )
        
        self.kernel2 = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, self.rank * in_channels)
        )
        
    def forward(self, v):
        height = v.size(-2)
        width = v.size(-1)
        coords = cell_centered_unit_grid((height, width), device=v.device).permute(1, 2, 0)
        left_kernel = self.kernel1(coords).reshape(height, width, self.out_channels, self.rank)
        right_kernel = self.kernel2(coords).reshape(height, width, self.rank, self.in_channels) 
        v = einops.einsum(right_kernel, v, 'h w r i, ... i h w -> ... r') / (height * width)
        v = einops.einsum(left_kernel, v, 'h w o r, ... r -> ... o h w')
        if self.bias is not None:
            v = v + self.bias
        return v

class LowRankInterpLinearOperator2d(LinearOperator2dBase):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 bias,
                 rank=32,
                 base_res=4,
                 mode='bilinear'):
        super().__init__(in_channels, out_channels, bias)        
        self.rank = rank
        self.base_res = base_res
        fan_in = self.in_channels * self.base_res * self.base_res
        self.k1 = nn.Parameter(torch.randn(
            self.rank,
            in_channels,
            self.base_res,
            self.base_res) * 1 / math.sqrt(fan_in))
        self.k2 = nn.Parameter(torch.randn(
            self.rank,
            out_channels,
            self.base_res,
            self.base_res) * 1 / math.sqrt(fan_in))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.mode = mode

    def _grid_sample(self, kernel, height, width):
        # grid_sample takes coords in [-1, 1]
        coords = 2 * cell_centered_unit_grid((height, width), device=kernel.device).permute(1, 2, 0) - 1
        coords = einops.repeat(coords, 'h w d -> b h w d', b=self.rank)
        # The documentation explaining this is not super clear, but align_corners=True
        # should essentially makes it so the kernel's boundary values actually
        # correspond to -1, 1 coordinates. This should make it so higher resolutions
        # do not extrapolate kernel values.
        return torch.nn.functional.grid_sample(kernel,
                                               coords, 
                                               align_corners=True, 
                                               mode=self.mode)

    def _kernel_with_interp(self, v):
        height = v.size(-2)
        width = v.size(-1)
        # The weight matrix is vertex-centered and includes boundary points. 
        # We need to evaluate at cell-centered coordinates. the kernel function is 
        # essentially determined as the interpolation of the weight matrices.
        k1 = self._grid_sample(self.k1, height, width)
        k2 = self._grid_sample(self.k2, height, width)
        v = einops.einsum(k1, v, 'r i h w, ... i h w -> ... r') / (height * width)
        v = einops.einsum(k2, v, 'r o h w, ... r -> ... o h w')
        return v

    def forward(self, v):
        # similar to FNO, a skip connection is needed since the low-rank
        # kernel explicitly projects to a finite space.
        v = self._kernel_with_interp(v) + seq_op(self.conv, v)
        if self.bias is not None:
            v = v + self.bias
        return v

class SplineInterpLinearOperator2d(LinearOperator2dBase):
    def __init__(self, in_channels, out_channels, bias, mode='bspline'):
        super().__init__(in_channels, out_channels, bias)        
        self.rank = 16
        self.base_res = 4
        fan_in = self.in_channels * self.base_res * self.base_res
        Spline = CubicBSplineGrid2d if mode == 'bspline' else CubicCatmullRomGrid2d
        
        # k1 is essentially a bilinear function. It's used in the integral,
        # so it doesn't really need to be smooth
        self.k1 = nn.Parameter(torch.randn(
            self.rank,
            in_channels,
            self.base_res,
            self.base_res) * 1 / math.sqrt(fan_in))
        
        # k2 is a spline function, so it's smoother than the bilinear function.
        self.k2 = Spline(resolution=(self.base_res, self.base_res),
                         n_channels=self.rank * self.out_channels)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        
    def _grid_sample(self, kernel, height, width):
        # grid_sample takes coords in [-1, 1]
        coords = 2 * cell_centered_unit_grid((height, width), device=kernel.device).permute(1, 2, 0) - 1
        coords = einops.repeat(coords, 'h w d -> b h w d', b=self.rank)
        # The documentation explaining this is not super clear, but align_corners=True
        # should essentially makes it so the kernel's boundary valeus actually
        # correspond to -1, 1 coordinates. This should make it so higher resolutions
        # do not extrapolate kernel values.
        return torch.nn.functional.grid_sample(kernel,
                                               coords, 
                                               align_corners=True, 
                                               mode='bilinear')
        
    def _interp_spline(self, spline, height, width, device):
        # spline takes coords in [0, 1]
        coords = cell_centered_unit_grid((height, width), device=device).permute(1, 2, 0)
        coords = einops.rearrange(coords, 'h w d -> (h w) d')
        interp = spline(coords)
        return einops.rearrange(interp, '(h w) ... -> h w ...', h=height, w=width)
        
    def _kernel_with_interp(self, v):
        height = v.size(-2)
        width = v.size(-1)
        # The weight matrix is vertex-centered and includes boundary points. 
        # We need to evaluate at cell-centered coordinates. the kernel function is 
        # essentially determined as the interpolation of the weight matrices.
        k1 = self._grid_sample(self.k1, height, width)
        k2 = self._interp_spline(self.k2, height, width, v.device)
        k2 = einops.rearrange(k2, 'h w (r o) -> r o h w', r=self.rank, o=self.out_channels)
        # do the neural operator stuff
        v = einops.einsum(k1, v, 'r i h w, ... i h w -> ... r') / (height * width)
        v = einops.einsum(k2, v, 'r o h w, ... r -> ... o h w')
        return v

    def forward(self, v):
        # similar to FNO, a skip connection is needed since the low-rank
        # kernel explicitly projects to a finite space.
        v = self._kernel_with_interp(v) + seq_op(self.conv, v)
        if self.bias is not None:
            v = v + self.bias
        return v


class AttentionLinearOperator2d(LinearOperator2dBase):
    r"""
    Technically not linear, since v is used in non-linear kernel func.
    But I am too lazy to change the name right now :-)
    This applies attention inside each subdomain. This is just
    a convenient/efficient way to implement an integral operator
    """
    def __init__(self, in_channels, out_channels, bias):
        super().__init__(in_channels, out_channels, bias)
        self.heads = 2
        self.in_channels = in_channels
    
        self.qk_head_dim = in_channels // self.heads
        self.v_head_dim = out_channels // self.heads
    
        self.qk = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.log_cpb = FixedCPB2d(in_channels, self.heads)
        
    def _qkv(self, v):
        v = einops.rearrange(v, 'b s c h w -> (b s) c h w')
        query, key = einops.rearrange(
            self.qk(v), 
            '... (split heads head_dim) h w -> split ... heads (h w) head_dim',
            split=2,
            heads=self.heads,
            head_dim=self.qk_head_dim)
        value = einops.rearrange(
            self.v(v), 
            '... (heads head_dim) h w -> ... heads (h w) head_dim',
            heads=self.heads,
            head_dim=self.v_head_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()
        
    def forward(self, v):
        batch_size = v.size(0)
        seq_len = v.size(1)
        height = v.size(-2)
        width = v.size(-1)
        
        query, key, value = self._qkv(v)
        pos_bias = self.log_cpb(height, width, device=v.device)
        attn = scaled_dot_product_attention(query, key, value, attn_mask=pos_bias)
        attn = einops.rearrange(attn, '(b s) heads (h w) head_dim -> b s (heads head_dim) h w',
                                b=batch_size,
                                s=seq_len,
                                h=height,
                                w=width,
                                heads=self.heads,
                                head_dim=self.v_head_dim)

        if self.bias is not None:
            attn = attn + self.bias
        return attn