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

        self.skip = nn.Conv2d(hc + 2, hc, kernel_size=(1, 1))

    def _apply_op(self, t, col_idx, row_idx, res_per_sub_x, res_per_sub_y):
        col_coords, row_coords = cell_centered_meshgrid(
                self.domain_size_x,
                self.domain_size_y,
                t.size(3),
                t.size(2))
        #coords = torch.stack((row_coords, col_coords), dim=0).to(t.device)
        #coords = coords.unsqueeze(0).repeat(t.size(0), 1, 1, 1)

