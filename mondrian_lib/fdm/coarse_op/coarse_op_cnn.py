import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLP
from mondrian_lib.fdm.get_subdomain_indices import get_subdomain_indices

class CoarseOpCNN(nn.Module):
    def __init__(
        self,
        min_res,
        in_channels,
        out_channels,
        hidden_channels,
        op_xlim,
        op_ylim):
        super().__init__()

        # TODO: min_res is currently unused
        self.min_res = min_res
        self.op_xlim = op_xlim
        self.op_ylim = op_ylim

        kernel_size = 11
        padding = (kernel_size - 1) // 2
        stride = 1

        self.conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        )

    def forward(self, t, global_xlim, global_ylim):
        assert t.dim() == 4
        x_res = t.size(3)
        y_res = t.size(2)

        res_per_x = x_res / global_xlim
        res_per_y = y_res / global_ylim

        # TODO: need to be careful...
        # Model is very sensitive to stride at different resolutions
        sample_stride = int((res_per_x / 64) * 8)

        h = t[:, :, ::sample_stride, ::sample_stride]
        h = self.conv(h)

        return F.interpolate(h, size=t.size()[-2:], mode='bilinear')
