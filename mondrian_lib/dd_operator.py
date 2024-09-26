import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn.norm import InstanceNorm
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch, pool_pos
from torch_geometric.utils import scatter
from torch_geometric.data import Batch, Data

from mondrian_lib.subdomain_utils import compute_clusters, subdomains_to_edge_index
from mondrian_lib.integral_op import subdomain_integral_op
from mondrian_lib.avg_pool import DDAvgPool

class DDNO(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels):
        super().__init__()

        mlp_hidden_channels = 128
        self.lift = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_channels),
            nn.GELU(),
            nn.Linear(mlp_hidden_channels, hidden_channels))

        self.project = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden_channels),
            nn.GELU(),
            nn.Linear(mlp_hidden_channels, out_channels))

        point_channels = 64
        self.point_encode = nn.Sequential(
            nn.Linear(2, mlp_hidden_channels),
            nn.GELU(),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels),
            nn.GELU(),
            nn.Linear(mlp_hidden_channels, point_channels))

        self.encoder1 = DDNO._block(point_channels + hidden_channels, hidden_channels, hidden_channels, 16, 16)
        self.encoder2 = DDNO._block(hidden_channels, 2 * hidden_channels, 2 * hidden_channels, 8, 8)
        self.encoder3 = DDNO._block(2 * hidden_channels, 4 * hidden_channels, 4 * hidden_channels, 4, 4)
        self.encoder4 = DDNO._block(4 * hidden_channels, 8 * hidden_channels, 8 * hidden_channels, 2, 2)

        self.pool1 = DDAvgPool(16, 16)
        self.pool2 = DDAvgPool(8, 8)
        self.pool3 = DDAvgPool(4, 4)
        self.pool4 = DDAvgPool(2, 2)

        self.bottleneck1 = DDNO._block(8 * hidden_channels, 8 * hidden_channels, 8 * hidden_channels, 1, 1)
        self.bottleneck2 = DDNO._block(8 * hidden_channels, 8 * hidden_channels, 8 * hidden_channels, 1, 1)

        self.decoder4 = DDNO._block(2 * (8 * hidden_channels), (8 * hidden_channels), 4 * hidden_channels, 2, 2)
        self.decoder3 = DDNO._block(2 * (4 * hidden_channels), (4 * hidden_channels), 2 * hidden_channels, 4, 4)
        self.decoder2 = DDNO._block(2 * (2 * hidden_channels), (2 * hidden_channels), hidden_channels, 8, 8)
        self.decoder1 = DDNO._block(2 * hidden_channels, hidden_channels, hidden_channels, 16, 16)
        self.decoder0 = DDNO._block(point_channels + 2 * hidden_channels, hidden_channels, hidden_channels, 16, 16)

    @staticmethod
    def _block(
            in_channels,
            hidden_channels,
            out_channels,
            n_subdomains_x,
            n_subdomains_y):
        args = 'x, src_coords, src_batch, tgt_coords, tgt_batch'
        return torch_geometric.nn.Sequential(args, [
            (DDOperator(in_channels,
                        hidden_channels,
                        hidden_channels,
                        dim=2,
                        n_subdomains_x=n_subdomains_x,
                        n_subdomains_y=n_subdomains_y), f'x, src_coords, src_batch, src_coords, src_batch -> x'),
            (InstanceNorm(hidden_channels), 'x, src_batch -> x'), 
            (nn.GELU(), 'x -> x'),
            (DDOperator(hidden_channels,
                        out_channels,
                        hidden_channels,
                        dim=2,
                        n_subdomains_x=n_subdomains_x,
                        n_subdomains_y=n_subdomains_y), f'x, src_coords, src_batch, tgt_coords, tgt_batch -> x'),
            (InstanceNorm(out_channels), 'x, tgt_batch -> x'), 
            (nn.GELU(), 'x -> x')
        ])

    def _encode(self, encoder, pooler, data):
        e = encoder(data.x,
                    data.pos,
                    data.batch,
                    data.pos,
                    data.batch)
        data_encoded = Batch(x=e, pos=data.pos, batch=data.batch)
        return pooler(data_encoded) 

    def _decode(self,
                decoder,
                input,
                skip,
                tgt_coords,
                tgt_batch):
        src_encode = self.point_encode(input.pos)
        tgt_encode = self.point_encode(tgt_coords)
        d = decoder(torch.cat((input.x, skip.x), dim=1),
                    input.pos, 
                    input.batch,
                    tgt_coords,
                    tgt_batch)
        data_decoded = Batch(x=d, pos=tgt_coords, batch=tgt_batch)
        return data_decoded

    def _bottleneck(self, bottleneck, input):
        src_encode = self.point_encode(input.pos)
        b = bottleneck(input.x,
                       input.pos,
                       input.batch,
                       input.pos,
                       input.batch)
        b = Batch(x=b, pos=input.pos, batch=input.batch)
        return b

    def forward(self, data):
        data.x = torch.cat((self.point_encode(data.pos), self.lift(data.x)), dim=1)

        p1 = self._encode(self.encoder1, self.pool1, data)
        p2 = self._encode(self.encoder2, self.pool2, p1)
        p3 = self._encode(self.encoder3, self.pool3, p2)
        p4 = self._encode(self.encoder4, self.pool4, p3)

        b = self._bottleneck(self.bottleneck1, p4)
        b = self._bottleneck(self.bottleneck2, b)

        d4 = self._decode(self.decoder4, b, p4, p3.pos, p3.batch)
        d3 = self._decode(self.decoder3, d4, p3, p2.pos, p2.batch)
        d2 = self._decode(self.decoder2, d3, p2, p1.pos, p1.batch)
        d1 = self._decode(self.decoder1, d2, p1, data.pos, data.batch)
        d0 = self._decode(self.decoder0, d1, data, data.pos, data.batch)

        data.x = self.project(d0.x)

        return data

class PointProjection(nn.Module):
    r"""
    Kernel function, projecting a point x = (x1, x2) to a matrix.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 dim):
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

class DDOperator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 dim,
                 n_subdomains_x,
                 n_subdomains_y):
        super().__init__()
 
        self.op_v = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels))
        self.op_tgt_kernel = nn.Sequential(
                nn.Linear(2 + hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, out_channels))

        self.n_subdomains_x = n_subdomains_x
        self.n_subdomains_y = n_subdomains_y

    def forward(self,
                x,
                src_coords,
                src_batch,
                tgt_coords,
                tgt_batch
    ):
        src_subdomains = compute_clusters(src_coords,
                                          src_batch,
                                          self.n_subdomains_x,
                                          self.n_subdomains_y) 
        tgt_subdomains = compute_clusters(tgt_coords,
                                          tgt_batch,
                                          self.n_subdomains_x,
                                          self.n_subdomains_y) 
        x, _ = subdomain_integral_op(x,
                                     src_subdomains,
                                     src_coords,
                                     tgt_subdomains=tgt_subdomains,
                                     tgt_coords=tgt_coords,
                                     op_v=self.op_v,
                                     op_tgt_out=self.op_tgt_out)
        return x
