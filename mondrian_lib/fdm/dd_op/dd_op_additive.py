import torch
import torch.nn as nn
from neuralop.layers.padding import DomainPadding
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.mlp import MLP
from mondrian_lib.fdm.get_subdomain_indices import get_subdomain_indices
from mondrian_lib.fdm.dd_op.dd_op_base import DDOpBase

class DDOpAdditive(DDOpBase):
    def __init__(
        self,
        layer,
        hc,
        op_xlim,
        op_ylim,
        overlap,
        domain_padding,
        use_coarse_op,
        use_padding
    ):
        super().__init__(layer, hc, op_xlim, op_ylim, overlap, domain_padding, use_coarse_op)

        self.padding = DomainPadding(domain_padding)
        self.use_padding = use_padding
        self.mlp_skip = MLP(hc, hc, hc, n_layers=1, n_dim=2)
        self.mlp_combine = MLP(2 * hc, hc, hc, n_layers=1, n_dim=2)

    def _apply_op(self, t, x_idx, y_idx, res_per_x, res_per_y, global_xlim, global_ylim):
        h = torch.zeros_like(t)
        mask = torch.zeros_like(t)
        for x in x_idx:
            for y in y_idx:
                h_in = t[:,:,y:y+res_per_y,x:x+res_per_x].clone()
                h_in_pad = self.padding.pad(h_in)
                h_in_pad = self.layer(h_in_pad)
                h_out = self.padding.unpad(h_in_pad)
                h[:,:,y:y+res_per_y,x:x+res_per_x] += h_out
                mask[:,:,y:y+res_per_y,x:x+res_per_x] += 1
        h_damp = h / mask
        c = self.c(t, global_xlim, global_ylim)
        h = torch.cat((h_damp, c), dim=1)
        h = self.mlp_combine(h)
        h = h + self.mlp_skip(t)
        return h
