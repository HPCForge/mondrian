import torch
import torch.nn as nn
from neuralop.layers.mlp import MLP
from mondrian_lib.fdm.integral import integral_2d

class FullRankLinearKernel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 num_filters):
        super().__init__()
        assert out_channels % num_filters == 0
        self.in_channels = in_channels
        self.out_channels = out_channels // num_filters
        self.num_filters = num_filters

        weight_size = (self.num_filters, self.out_channels, self.in_channels, 2)
        bias_size = (self.num_filters, self.out_channels, self.in_channels)

        self.Wx = torch.empty(weight_size, requires_grad=True)
        torch.nn.init.kaiming_normal(self.Wx)

        self.Wy = torch.empty(weight_size, requires_grad=True)
        torch.nn.init.kaiming_normal(self.Wy)

        self.B = torch.empty(bias_size, requires_grad=True)
        torch.nn.init.kaiming_normal(self.B)

    def forward(self, v):
        r"""
        Integral kernel operator. For each coordinate, m we evaluate the
        operator to all discretization points of v.
        Args:
            v: discretized function [batch, in_size, H, W]
        Returns
            u: discretized function [batch, out_size, H, W]
        """

        x_res = v.size(3)
        y_res = v.size(2)

        # Just use [-1, 1], since we can assume arbitrary coord system
        x_coords, y_coords = torch.meshgrid(
                torch.linspace(-1, 1, x_res, device=v.device),
                torch.linspace(-1, 1, y_res, device=v.device),
                indexing='xy')

        # [H, W, 2]
        x = torch.stack((y_coords, x_coords), dim=-1)

        # [H, W, num_filters, out_size, in_size]
        Wx = torch.einsum('foid,hwd->hwfoi', self.Wx, x)
        BWx = self.B.unsqueeze(0).unsqueeze(0) + Wx
        
        # [H, W, out_size, in_size]
        Wy = torch.einsum('foid,hwd->hwfoi', self.Wy, x)

        # [batch, H, W, in_size]
        vp = v.permute(0, 2, 3, 1)

        dx = x_coords[0, 1] - x_coords[0, 0]
    
        # Integrate vp
        # [batch, in_size]
        integral_vp = integral_2d(vp, dx=dx, dim1=1, dim2=2)

        # Compute [B + Wx]Iy
        # [batch, H, W, filter_size, out_size]
        scaled_integral_vp = torch.einsum('hwfoi,bi->bhwfo', BWx, integral_vp)

        # integrate vp weighted by Wy
        Wyvp = torch.einsum('hwfoi,bhwi->bhwfo', Wy, vp)
        # [batch, filter_size, out_size]
        integral_weighted_vp = integral_2d(Wyvp, dx=dx, dim1=1, dim2=2)

        # [batch, H, W, filter_size, out_size] 
        up = scaled_integral_vp + integral_weighted_vp.unsqueeze(1).unsqueeze(1)
        # [batch, H, W, filter_size * out_size]
        up = torch.flatten(up, start_dim=-2)

        return up.permute(0, 3, 1, 2)
