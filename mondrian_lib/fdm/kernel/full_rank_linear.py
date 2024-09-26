import torch
import torch.nn as nn
from neuralop.layers.mlp import MLP
from mondrian_lib.fdm.integral import integral_2d

class FullRankLinearKernel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_filters):
        super().__init__()
        assert out_channels % num_filters == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters 

        weight_size = (self.num_filters, self.out_channels, self.in_channels, 2)
        self.Wx = nn.Parameter(torch.empty(weight_size))
        self.Wy = nn.Parameter(torch.empty(weight_size))
        torch.nn.init.kaiming_uniform_(self.Wx)
        torch.nn.init.kaiming_uniform_(self.Wy)

        self.to_out = nn.Linear(num_filters * self.out_channels, self.out_channels)

    def forward(self, v, v_coords, target_coords):
        r"""
        Integral kernel operator. For each coordinate, m we evaluate the
        operator to all discretization points of v.
        Args:
            v: discretized function [batch, in_size, H, W]
            v_coords: coordinates for v. [H, W, D]
            target_coords: coordinates to map to [H2, W2, D]
        Returns
            u: discretized function [batch, out_size, H, W]
        """

        x_res = v.size(3)
        y_res = v.size(2)

        # [H, W, num_filters, out_size, in_size]
        Wx = torch.einsum('foid,hwd->hwfoi', self.Wx, target_coords)
        
        # [H, W, out_size, in_size]
        Wy = torch.einsum('foid,hwd->hwfoi', self.Wy, v_coords)

        # [batch, H, W, in_size]
        vp = v.permute(0, 2, 3, 1)

        dx = v_coords[0, 1, 0] - v_coords[0, 0, 0]
    
        # Integrate vp
        # [batch, in_size]
        integral_vp = integral_2d(vp, dx=dx, dim1=1, dim2=2)

        # Compute Wx * Iy
        # [batch, H, W, filter_size, out_size]
        scaled_integral_vp = torch.einsum('hwfoi,bi->bhwfo', Wx, integral_vp)

        # integrate over vp weighted by Wy
        Wyvp = torch.einsum('hwfoi,bhwi->bhwfo', Wy, vp)
        # [batch, filter_size, out_size]
        integral_weighted_vp = integral_2d(Wyvp, dx=dx, dim1=1, dim2=2)

        # [batch, H, W, filter_size, out_channels] 
        up = scaled_integral_vp + integral_weighted_vp.unsqueeze(1).unsqueeze(1)
        
        up = up.sum(3)
        
        # [batch, H, W, filter_size * out_channels]
        #up = torch.flatten(up, start_dim=-2)
        # [batch, H, W, out_channels]
        #up = self.to_out(up)

        return up.permute(0, 3, 1, 2)
