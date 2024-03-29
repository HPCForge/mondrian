import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLP
from mondrian_lib.models.fdm.get_subdomain_indices import get_subdomain_indices
from mondrian_lib.models.fdm.cell_centered_coords import cell_centered_meshgrid

class CoarseOpGNN(nn.Module):
    def __init__(
        self,
        min_res,
        in_channels,
        out_channels,
        hidden_channels,
        op_xlim,
        op_ylim
    ):
        super().__init__()

        self.min_res = min_res
        assert self.min_res % 2 == 0
        self.op_xlim = op_xlim
        self.op_ylim = op_ylim

    def _edge_weighted_graph(t, global_xlim, global_ylim, x_res, y_res):
        x_coords, y_coords = cell_centered_meshgrid(
                global_xlim, global_ylim, x_res, y_res)

        # pick indices uniformly at random, number scales with domain size
        num_samples = 5 * global_xlim * global_ylim
        x_indices = torch.arange(x_res).multinomial(num_samples=num_samples, replacement=True)
        y_indices = torch.arange(y_res).multinomial(num_samples=num_samples, replacement=True)
        coord_indices = torch.stack((x_indices, y_indices), dim=0)

        # node value is the vector at coordinate
        x = t[:, :, x_indices, y_indices]

        # create edges between coordinate in a subdomain

        # create edges between subdomains.

    def forward(self, t, global_xlim, global_ylim):
        assert t.dim() == 4
        x_res = t.size(3)
        y_res = t.size(2)

        res_per_x = x_res / global_xlim
        res_per_y = y_res / global_ylim



