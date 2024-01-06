import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, SSGConv, ChebConv, PointNetConv
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn.models import GAT, GCN
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
from data.hdf5_dataset import HDF5Dataset
from ddm.partition import partition_with_overlap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mfem.ser as mfem
import ctypes

from scipy.interpolate import griddata

class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.l1 = nn.Linear(in_size, out_size)
        self.l2 = nn.Linear(out_size, out_size)

    def forward(self, x):
        x = self.l1(x)
        x = F.gelu(x)
        return self.l2(x)

class ChebResidual(torch.nn.Module):
    def __init__(self, embed_size, K):
        super().__init__()
        self.c1 = ChebConv(embed_size, embed_size, K=5)
        self.c2 = ChebConv(embed_size, embed_size, K=5)
        self.l1 = nn.Linear(embed_size, embed_size)

    def F(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        tmp = F.gelu(self.c1(data.x, edge_index, edge_attr, batch=data.batch))
        return self.c2(tmp, edge_index, edge_attr, batch=data.batch)

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        return F.gelu(self.F(data) + self.l1(data.x))

class InwardSolve(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_size = 4
        embed_size = 64
        self.node_embed = MLP(in_size, embed_size)
        self.node_down = MLP(embed_size, 1)
        self.c3 = ChebResidual(embed_size, K=5)
        self.c4 = ChebResidual(embed_size, K=5)
        self.c5 = ChebResidual(embed_size, K=3)
        self.c6 = ChebResidual(embed_size, K=3)
        
    def get_1hop_subgraph(self, data, node_index):
        extended_node_index, edge_index, _, edge_mask = \
                torch_geometric.utils.k_hop_subgraph(node_index, 1, data.edge_index)
        return extended_node_index, Data(x=data.x,
                                         edge_index=edge_index,
                                         edge_attr=data.edge_attr[edge_mask])

    def forward(self, data):
        data.x = F.gelu(self.node_embed(data.x))
        
        # start from boundary, and work inwards
        #node_index = data.boundary_indices
        #node_index, subgraph = self.get_1hop_subgraph(data, node_index)
        #data.x = F.gelu(self.c1(subgraph.x, subgraph.edge_index, subgraph.edge_attr)) + data.x 
        #node_index, subgraph = self.get_1hop_subgraph(data, node_index)
        #data.x = F.gelu(self.c2(subgraph.x, subgraph.edge_index, subgraph.edge_attr)) + data.x

        #original = data.x.copy()

        data.x = self.c3(data)
        data.x = self.c4(data)
        data.x = self.c5(data)
        #data.x = self.c6(data)

        data.x = self.node_down(data.x)
        return data.x

class RandomEdges(torch_geometric.transforms.BaseTransform):
    r"""
    Adds additional edges pointing out from boundary nodes
    """
    def __init__(self, num_edges_to_add):
        super().__init__()
        self.num_edges_to_add = num_edges_to_add

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        num_bdy_nodes = data.boundary_indices.size(0)
    
        # assumes that boundary nodes are at the front of data tensors
        source_indices = torch.randint(0, num_bdy_nodes, (1, self.num_edges_to_add), dtype=torch.long)
        target_indices = torch.randint(num_bdy_nodes, num_nodes, (1, self.num_edges_to_add), dtype=torch.long)
        edge_index_to_add = torch.cat((source_indices, target_indices), dim=0)
        edge_index_to_add = torch_geometric.utils.to_undirected(edge_index_to_add)
        data.edge_index = torch.cat([data.edge_index, edge_index_to_add], dim=1)

        return data

class FixBoundaryEdges(torch_geometric.transforms.BaseTransform):
    r"""
    Remove edges pointing towards boundary nodes.
    """
    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        num_bdy_nodes = data.boundary_indices.size(0)

        target = data.edge_index[1]
        t = target == data.boundary_indices.unsqueeze(-1)
        edge_mask = t.sum(dim=0) != 0
        data.edge_index = data.edge_index[:, ~edge_mask]
        data.edge_attr = data.edge_attr[~edge_mask]
        return data

def main():
    filename = 'datagen/ng_solve/poisson.hdf5'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 16

    # weight by inverse distance, and remove edges between very close nodes
    transform = torch_geometric.transforms.Compose([
        torch_geometric.transforms.KNNGraph(),
        torch_geometric.transforms.ToUndirected(),
        torch_geometric.transforms.RemoveDuplicatedEdges(),
        torch_geometric.transforms.Distance(),
        FixBoundaryEdges(),
    ])
    dataset = HDF5Dataset(filename, transform=transform)

    #sample = dataset[0]
    #data, cluster = partition_with_overlap(sample, 4, False)
    #
    #plt_grid(data, cluster.float())
    #
    #return

    train_size = int(0.7 * len(dataset))
    test_and_val_size = len(dataset) - train_size
    test_size = int(0.5 * test_and_val_size)
    val_size = int(0.5 * test_and_val_size)

    train_dataset, val_dataset, test_dataset = \
            torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = InwardSolve().float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print('HOPS: ', torch_geometric.utils.get_num_hops(model))

    print('dataset size: ', len(dataset))
    print(train_dataset[0].edge_attr.size())

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            optimizer.zero_grad()
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            print('train mse: ', loss)

        model.eval()
        for batch in val_loader:
            with torch.no_grad():
                batch = batch.to(device)
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y)
            print('val mse: ', loss)

    model.eval()
    accum_test_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y, reduction='mean')
            print('test mse: ', loss)
            accum_test_loss += F.mse_loss(pred, batch.y, reduction='sum')
    test_mse = accum_test_loss / (len(test_loader) * batch_size)
    print('total test mse: ', test_mse) 
            
    data = test_dataset[2]
    data = data.to(device)
    pred = model(data)

    plt_grid(data.cpu(), pred.detach().cpu())
    plt_scatter(data.cpu(), pred.detach().cpu())

def build_grid(arr, pos, res=50):
    mins = pos.min(dim=0).values
    maxs = pos.max(dim=0).values
    gridx, gridy, gridz = np.meshgrid(np.linspace(mins[0], maxs[0], res),
                                      np.linspace(mins[1], maxs[1], res),
                                      np.linspace(mins[2], maxs[2], res))
    grid = griddata(pos.numpy(), arr.numpy(), (gridx, gridy, gridz), method='linear')
    return grid

def plt_grid(data, pred):
    n_cols = 5
    fig, axarr = plt.subplots(2, n_cols)

    sol_grid = build_grid(data.y, data.pos)
    pred_grid = build_grid(pred, data.pos)

    for idx, c in enumerate(range(0, sol_grid.shape[0], sol_grid.shape[0] // n_cols)):
        print(c, sol_grid.shape)
        axarr[0, idx].imshow(sol_grid[c].squeeze())
        axarr[1, idx].imshow(pred_grid[c].squeeze())

    plt.savefig('gnn_viz.png')

def plt_scatter(data, pred):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = data.pos[:, 0]
    y = data.pos[:, 1]
    z = data.pos[:, 2]
    ax.scatter(x, y, z, c=data.y)
    plt.savefig('gnn_scat.png')



if __name__ == '__main__':
    main()
