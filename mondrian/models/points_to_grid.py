import torch
from torch import nn

from torch_cluster import grid_cluster

class PointsToGrid(nn.Module):
    def __init__(self, subdomain_size_x, subdomain_size_y):
        self.subdomain_size_x = subdomain_size_x
        self.subdomain_size_y = subdomain_size_y
        
    def forward(self, 
                values, 
                pos, 
                batch,
                domain_size_x,
                domain_size_y):
        # Notes on torch_cluster.grid_cluster, since it is not well documented.
        #   1.
        #   2.
        clusters = grid_cluster(pos,
                                batch=batch,
                                size=(self.subdomain_size_x, self.subdomain_size_y),
                                # TODO: figure this out... Needs to contain the `pos` values
                                # this is needed so the subdomain locations are consistent with different inputs
                                start=(0, 0),
                                end=(domain_size_x, domain_size_y))
        