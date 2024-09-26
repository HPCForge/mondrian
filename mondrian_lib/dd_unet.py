from typing import Callable, List, Union, Tuple

import torch
from torch import Tensor
from torch import nn

import torch_geometric
from torch_geometric.nn import GCNConv, MLP, knn_interpolate
from torch_geometric.nn.norm import InstanceNorm
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.data import Batch
from torch_geometric.utils.repeat import repeat
from torch_geometric.utils import dropout_edge

from mondrian_lib.avg_pool import DDAvgPool
from mondrian_lib.subdomain_utils import compute_clusters, subdomains_to_edge_index
from mondrian_lib.integral_op import subdomain_integral_to_reference

class DDGraphUNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        subdomain_size: List[Tuple[int, int]],
    ):
        super().__init__()
        assert len(subdomain_size) >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = len(subdomain_size)
        self.subdomain_size = subdomain_size
        self.act = nn.GELU()

        channels = hidden_channels

        self.lift = nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.GELU(),
                nn.Linear(256, channels),
                nn.GELU())

        self.project = nn.Sequential(
                nn.Linear(channels, 256),
                nn.GELU(),
                nn.Linear(256, out_channels))

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_norms = torch.nn.ModuleList()
        
        self.down_convs.append(DDOpGNN(channels,
                                       channels,
                                       channels,
                                       self.subdomain_size[0][0],
                                       self.subdomain_size[0][1]))
        self.down_norms.append(InstanceNorm(channels))
        for i in range(self.depth):
            self.pools.append(DDAvgPool(self.subdomain_size[i][0], self.subdomain_size[i][1]))
            self.down_convs.append(DDOpGNN(channels,
                                           channels,
                                           channels,
                                           self.subdomain_size[i][0],
                                           self.subdomain_size[i][1]))
            self.down_norms.append(InstanceNorm(channels))

        self.bottleneck1 = DDOpGNN(channels,
                                  channels,
                                  channels,
                                  1,
                                  1)
        self.bottleneck2 = DDOpGNN(channels,
                                  channels,
                                  channels,
                                  1,
                                  1)
        self.bottleneck_norm1 = InstanceNorm(channels)
        self.bottleneck_norm2 = InstanceNorm(channels)

        self.up_convs = torch.nn.ModuleList()
        self.op_convs = torch.nn.ModuleList()
        self.up_norms = torch.nn.ModuleList()
        self.op_norms = torch.nn.ModuleList()
        for i in range(self.depth):
            j = self.depth - 1 - i
            self.up_convs.append(FPModule(3, MLP([2 * channels, channels, channels])))
            self.op_convs.append(DDOpGNN(channels,
                                         channels,
                                         channels,
                                         subdomain_size[j][0],
                                         subdomain_size[j][1]))
            self.up_norms.append(InstanceNorm(channels))
            self.op_norms.append(InstanceNorm(channels))

    def forward(self, data) -> Tensor:
        data_in = data.clone()

        data.x = self.lift(data.x)

        x = self.down_convs[0](data.x, data.pos, data.batch)
        x = self.down_norms[0](x)
        x = self.act(x)
        data.x = x

        data_history = [data]

        for i in range(1, self.depth + 1):
            pooled_data = self.pools[i - 1](data)
            x = self.down_convs[i](pooled_data.x,
                                   pooled_data.pos,
                                   pooled_data.batch)
            x = self.act(self.down_norms[i](x))
            pooled_data.x = x

            if i < self.depth:
                data_history += [pooled_data]
        
            data = pooled_data

        h = self.bottleneck1(data.x,
                             data.pos,
                             data.batch,
                             dropout=False)
        h = self.act(self.bottleneck_norm1(h))
        #h = self.bottleneck1(h + data.x,
        #                     data.pos,
        #                     data.batch,
        #                     dropout=False)
        #h = self.act(self.bottleneck_norm2(h))
        data.x = data.x + h

        for i in range(self.depth):
            j = self.depth - 1 - i

            skip_data = data_history[j]

            x, _, _ = self.up_convs[i](data.x,
                                       data.pos,
                                       data.batch,
                                       skip_data.x,
                                       skip_data.pos,
                                       skip_data.batch)
            #x = self.up_convs[i](data.x,
            #                     data.pos,
            #                     data.batch,
            #                     skip_data.x,
            #                     skip_data.pos,
            #                     skip_data.batch)
            x = self.up_norms[i](x)
            x = self.act(x)

            #data = Batch(x=torch.cat((x, skip_data.x), dim=1),
            #             pos=skip_data.pos,
            #             batch=skip_data.batch)
            data = Batch(x=x + skip_data.x,
                         pos=skip_data.pos,
                         batch=skip_data.batch)
            x = self.op_convs[i](data.x,
                                 data.pos,
                                 data.batch)
            if i < self.depth - 1:
                x = self.act(self.up_norms[i](x))

            data = Batch(x=x, pos=skip_data.pos, batch=skip_data.batch)

        data.x = self.project(data.x)

        return data

class DDOpGNN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 n_subdomains_x,
                 n_subdomains_y):
        super().__init__()
 
        self.n_subdomains_x = n_subdomains_x
        self.n_subdomains_y = n_subdomains_y

        self.input_encode = nn.Sequential(
                nn.Linear(in_channels + 2, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 2 * hidden_channels))

        self.skip = nn.Linear(in_channels + 2, out_channels, bias=False)

        self.layer = torch_geometric.nn.GraphConv(2 * hidden_channels,
                                                  out_channels,
                                                  aggr='mean')

    def reset_parameters(self):
        pass
        #self.layer.reset_parameters()

    def forward(self,
                x,
                src_coords,
                src_batch,
                dropout=False):
        src_subdomains = compute_clusters(src_coords,
                                          src_batch,
                                          self.n_subdomains_x,
                                          self.n_subdomains_y) 
        node_ids = torch.arange(x.size(0), device=x.device)
        edge_index = subdomains_to_edge_index(node_ids, src_subdomains)

        if dropout:
            edge_index = dropout_edge(edge_index, p=0.5, training=self.training)[0]

        edge_weight = torch.linalg.vector_norm(src_coords[edge_index[0]] - src_coords[edge_index[1]],
                                               dim=1)
        input = torch.cat((x, src_coords), dim=1)
        x_enc = self.input_encode(input)
        x = self.skip(input) + self.layer(x_enc, edge_index, edge_weight)
        return x

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class DDOpGNNUpsample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 n_subdomains_x,
                 n_subdomains_y):
        super().__init__()
 
        self.n_subdomains_x = n_subdomains_x
        self.n_subdomains_y = n_subdomains_y

        #self.input_encode = nn.Sequential(
        #        nn.Linear(in_channels + 2, hidden_channels),
        #        nn.GELU(),
        #        nn.Linear(hidden_channels, 2 * hidden_channels))

        self.input_encode = nn.Sequential(in_channels + 2, hidden_channels)

        self.skip = nn.Linear(in_channels, out_channels, bias=False)

        self.layer = torch_geometric.nn.GraphConv(2 * hidden_channels,
                                                  out_channels,
                                                  aggr='mean')
        
    def forward(self,
                src_node_values,
                src_coords,
                src_batch,
                tgt_node_values,
                tgt_coords,
                tgt_batch):
        src_subdomains = compute_clusters(src_coords,
                                          src_batch,
                                          self.n_subdomains_x,
                                          self.n_subdomains_y) 
        tgt_subdomains = compute_clusters(tgt_coords,
                                          tgt_batch,
                                          self.n_subdomains_x,
                                          self.n_subdomains_y) 

        subdomains = torch.cat((src_subdomains, tgt_subdomains))
        batch = torch.cat((src_batch, tgt_batch))
        pos = torch.cat((src_coords, tgt_coords), dim=0)
        node_values = torch.cat((src_node_values, tgt_node_values), dim=0)
        node_values = self.input_encode(torch.cat((node_values, pos), dim=1))

        src_node_ids = torch.arange(src_node_values.size(0), device=src_node_values.device)
        node_ids = torch.arange(node_values.size(0), device=src_node_values.device)
        edge_index = subdomains_to_edge_index(node_ids, subdomains)

        edge_weight = torch.linalg.vector_norm(pos[edge_index[0]] - pos[edge_index[1]],
                                               dim=1)
        edge_index, edge_weight = torch_geometric.utils.remove_self_loops(edge_index, edge_weight)

        # only get tgt values, which are all nodes after the src nodes.
        out_node_values = self.layer(node_values, edge_index, edge_weight)
        tgt_values = self.skip(tgt_node_values) + out_node_values[src_coords.size(0):]

        return tgt_values

class PointProjection(nn.Module):
    r"""
    Kernel function, projecting a point x = (x1, x2) to a matrix.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = nn.Sequential(
                nn.Linear(dim, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, out_channels * in_channels))

    def forward(self, x):
        x = self.op(x)
        return torch.unflatten(x, dim=1, sizes=(self.out_channels, self.in_channels))


class ToReference(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 n_subdomains_x,
                 n_subdomains_y):
        super().__init__()
        self.n_subdomains_x = n_subdomains_x
        self.n_subdomains_y = n_subdomains_y
        self.n_subdomains = n_subdomains_x * n_subdomains_y

        # physical size of reference domain that subdomains are mapped to
        self.x_size = -1 / n_subdomains_x, 1 / n_subdomains_x
        self.y_size = -1 / n_subdomains_y, 1 / n_subdomains_y

        #self.op_v = nn.Sequential(
        #        nn.Linear(in_channels, hidden_channels),
        #        nn.GELU(),
        #        nn.Linear(hidden_channels, hidden_channels))

        self.op_src_kernel = PointProjection(hidden_channels, out_channels, hidden_channels)
        self.op_tgt_kernel = PointProjection(hidden_channels, out_channels, hidden_channels)


    def _tgt_coords(self,
                    subdomain_xres,
                    subdomain_yres,
                    batch_size,
                    device):
        r"""
        Each subdomain in each batch will map to a reference grid in [-1, 1]^2
        """
        x = torch.linspace(*self.x_size, subdomain_xres, device=device)
        y = torch.linspace(*self.y_size, subdomain_yres, device=device)
        # [H, W, 2]
        tgt_coords = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
        # [H * W, 2]
        tgt_coords = torch.flatten(tgt_coords, end_dim=1) 
        return tgt_coords

    def forward(self,
                v,
                src_coords,
                src_batch):
        src_subdomains, size_x, size_y = compute_clusters(src_coords,
                                                          src_batch,
                                                          self.n_subdomains_x,
                                                          self.n_subdomains_y) 

        # number of points in each subdomain
        _, res = torch.unique(src_subdomains, return_counts=True)
        # since we want things to be passed via a tensor to FNO,
        # we need each subdomain for each batch entry to use
        # the same resolution.
        tgt_res = int(torch.max(res))

        batch_size = src_batch.max() + 1
        tgt_coords = self._tgt_coords(tgt_res, tgt_res, batch_size, v.device)

        # [B, S, J, m]
        u = subdomain_integral_to_reference(
                self.n_subdomains,
                v,
                src_coords,
                src_subdomains,
                src_batch=src_batch,
                tgt_coords=tgt_coords,
                op_src_kernel=self.op_src_kernel,
                op_tgt_kernel=self.op_tgt_kernel)

        # [B, S, H, W, m]
        u = torch.unflatten(u,
                            dim=2,
                            sizes=(tgt_res, tgt_res))

        return u

if __name__ == '__main__':
    r = ToReference(1, 10, 1, 4, 4)
    res = 10
    x = torch.linspace(-1, 1, res)
    y = torch.linspace(-1, 1, res)
    coords = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1).flatten(end_dim=1)
    v = torch.sin(coords[:, 0] + coords[:, 1]).unsqueeze(1)
    batch = torch.zeros(v.size(0))
    r(v, coords, batch)
