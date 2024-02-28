"""
Name: Alex Danielian
Date: 29 Jan 2024
Description:    This file contains the main training loop for the torchdeq model, and the model itself.
                
"""

import torch
import torchdeq
from torchdeq.core import get_deq
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TAGConv, ChebConv
from torch_geometric.loader import DataLoader
from data.hdf5_dataset import HDF5Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mfem.ser as mfem
import ctypes

from scipy.interpolate import griddata

debug = False

class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.l1 = nn.Linear(in_size, out_size)
        self.l2 = nn.Linear(out_size, out_size)

    def forward(self, x):
        # If x is a tuple. then unpack and take the first element
        if isinstance(x, tuple):
            x = x[0][0]
        
        if debug:
            print("MLP Forward(): ")
            print("\tIn:Out ", self.in_size, ":",self.out_size)
            print("\tX: ", x.shape)

        x = self.l1(x)
        x = F.gelu(x)
        x = self.l2(x)
        return x

"""
Node Regression: Predict from the boundary conditions.
"""
class GNN(torch.nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super().__init__()
        self.conv1 = TAGConv(in_size, hidden)
        self.conv2 = TAGConv(hidden, hidden)
        self.conv3 = TAGConv(hidden, out_size)

    def forward(self, data):
        # Data is a tensor here???
        x, edge_index, edge_attr, pos = data
        # apply the graph convolutional layers

        if debug:
            print("GNN Forward(): ")
            print("\tX: ", x.shape)
            print("\tEdge Index: ", edge_index.shape)
            print("\tEdge Weight: ", edge_attr.shape)
            print("\tPos: ", pos.shape)

        x = torch.cat([x, pos], dim=1)
        if debug: print("\tX: ", x.shape)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        if debug: print("Conv1: ", x.shape)

        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        if debug: print("Conv2: ", x.shape)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.log_softmax(x, dim=1) 
        if debug: print("Conv3: ", x.shape)

        return x


class PoissonDEQ(nn.Module):
    """
    Description:    DEQ Model for Poisson Equation (poisson.hdf5 data)
                    Using the torchdeq library, this model is able to solve the Poisson equation
                    using a neural network. The model is trained on the interior of the domain,
                    and the boundary conditions are applied after the model has converged.
    """
    def __init__(self):
        super().__init__()
        
        """
        .   Model Architecture [Poisson DEQ]:   1. MLP transform up from input into embedded
        .                                       2. GNN implementation of DEQ to solve for z_star
        .                                       3. MLP transform down from embedded to output
        .                                       
        """ 
        self.ins = 4
        self.emb = 64
        self.hidden = 128
        self.outs = 1
        self.mlpup = MLP(self.ins, self.emb)
        self.gnn = GNN(self.emb + 3, self.hidden, self.emb)
        self.mlpdown = MLP(self.emb, self.outs)
        self.deq = get_deq(self.gnn)

    def forward(self, data):
        z = self.mlpup(data.x)

        if debug:
            print("PoissonDEQ Forward(): ")
            print("\tZ: ", z.shape)
            print("\tX: ", data.x.shape)
            print("\tEdge Index: ", data.edge_index.shape)
            print("\tEdge Weight: ", data.edge_attr.shape)
            print("\tPos: ", data.pos.shape)

        def func(_z):
            return self.gnn((_z, data.edge_index, data.edge_attr, data.pos))
        
        if debug: print("\tself.deq()")
        z_star = self.deq(func, z)

        if debug: print("\tZ*: ", z_star.shape)
        z = self.mlpdown(z_star)
        if debug: print("\tZ: ", z.shape)
        return z, z_star

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

    filename = "datagen/ng2000/poisson.hdf5"
    device = 'cude:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 1

    print("Creating Transform...")

        # weight by inverse distance, and remove edges between very close nodes
    transform = torch_geometric.transforms.Compose([
        torch_geometric.transforms.KNNGraph(),
        torch_geometric.transforms.ToUndirected(),
        torch_geometric.transforms.RemoveDuplicatedEdges(),
        torch_geometric.transforms.Distance(),
        FixBoundaryEdges(),
    ])

    print("Creating Dataset...")

    dataset = HDF5Dataset(filename, transform=transform)
    
    train_size = int(0.7 * len(dataset))
    test_and_val_size = len(dataset) - train_size
    test_size = int(0.5 * test_and_val_size)
    val_size  = int(0.5 * test_and_val_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    print("Creating Model...")

    weight_decay = 1e-5
    model = PoissonDEQ().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.00001)

    print('HOPS: ', torch_geometric.utils.get_num_hops(model))

    print('Dataset size: ', len(dataset))
    print(train_dataset[0].edge_attr.size())

    print("Training Model...")
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            optimizer.zero_grad()
            pred_tensor = pred[0]
            loss = F.mse_loss(pred_tensor, batch.y)
            loss.backward()
            optimizer.step()
            print('train mse: ', loss)

        model.eval()
        for batch in val_loader:
            with torch.no_grad():
                batch = batch.to(device)
                pred = model(batch)
                pred_tensor = pred[0]
                loss = F.mse_loss(pred_tensor, batch.y)
                #loss = CentroidSensitiveLoss(pred_tensor, batch.y, CentroidOf(batch), RadiusOf(batch))
            print('val mse: ', loss)
        lr_scheduler.step()

    print("Testing Model...")
    model.eval()
    accum_test_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch)
            pred_tensor = pred[0]
            loss = F.mse_loss(pred_tensor, batch.y)
            #loss = CentroidSensitiveLoss(pred_tensor, batch.y, CentroidOf(batch), RadiusOf(batch))
            print('test mse: ', loss)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_mse': loss
    }, 'poisson_model.pth')

    data = test_dataset[2]
    data = data.to(device)
    pred = model(data)
    pred_tensor = pred[0]
    pred_stat = pred[1]
    plt_grid(data.cpu(), pred_tensor.detach().cpu())
    plt_scatter(data.cpu(), pred_tensor.detach().cpu())

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