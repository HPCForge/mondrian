import torch
import torch.nn as nn
from neuralop.layers.padding import DomainPadding
from neuralop.layers.mlp import MLP
from mondrian_lib.fdm.dd_op.dd_op_base import DDOpBase
from mondrian_lib.fdm.cell_centered_coords import cell_centered_meshgrid

class DDOpAdditive(DDOpBase):
    def __init__(
        self,
        layer,
        hc,
        domain_size_x,
        domain_size_y,
        subdomain_size_x,
        subdomain_size_y,
        overlap,
        domain_padding,
        use_coarse_op,
        use_padding
    ):
        super().__init__(layer,
                         hc,
                         domain_size_x,
                         domain_size_y,
                         subdomain_size_x,
                         subdomain_size_y,
                         overlap,
                         domain_padding,
                         use_coarse_op)

        self.padding = DomainPadding(domain_padding)
        self.use_padding = use_padding

    def _apply_op(self, t, x_idx, y_idx, res_per_x, res_per_y):
        x_coords, y_coords = cell_centered_meshgrid(
                self.domain_size_x,
                self.domain_size_y,
                t.size(3),
                t.size(2))
        coords = torch.stack((y_coords, x_coords), dim=-1).to(t.device)

        h = torch.zeros_like(t)
        mask = torch.zeros_like(t)
        for x in x_idx:
            for y in y_idx:
                h_in = t[:,:,y:y+res_per_y,x:x+res_per_x].clone()
                coords_in = coords[y:y+res_per_y,x:x+res_per_x].clone()
                h_out = self.layer(h_in, coords_in)
                h[:,:,y:y+res_per_y,x:x+res_per_x] += h_out
                mask[:,:,y:y+res_per_y,x:x+res_per_x] += 1
        h_damp = h / mask
        return h_damp
