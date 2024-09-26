r"""
Taken and modified from pytorch geometric:
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/avg_pool.html#avg_pool
- This just removes the edge pooling, which is non-optional in their implementation.
"""

from typing import Callable, Optional, Tuple

from torch import Tensor
import torch.nn as nn

from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.utils import add_self_loops, scatter

from mondrian_lib.subdomain_utils import compute_clusters

class DDAvgPool(nn.Module):
    def __init__(self,
                 n_subdomains_x,
                 n_subdomains_y):
        super().__init__()
        self.n_subdomains_x = n_subdomains_x
        self.n_subdomains_y = n_subdomains_y

    def forward(self, data):
        src_subdomains = compute_clusters(data.pos,
                                          data.batch,
                                          self.n_subdomains_x,
                                          self.n_subdomains_y) 
        data = avg_pool(src_subdomains, data)
        return data


def _avg_pool_x(
    cluster: Tensor,
    x: Tensor,
    size: Optional[int] = None,
) -> Tensor:
    return scatter(x, cluster, dim=0, dim_size=size, reduce='mean')

def avg_pool(
    cluster: Tensor,
    data: Data,
    transform: Optional[Callable] = None,
) -> Data:
    cluster, perm = consecutive_cluster(cluster)

    x = None if data.x is None else _avg_pool_x(cluster, data.x)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    data = Batch(batch=batch, x=x, pos=pos)

    return data

