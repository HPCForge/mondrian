import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, PointNetConv
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn.models import GAT, GCN
from data.hdf5_dataset import HDF5Dataset
import networkx as nx
import matplotlib
import matplotlib.pyplot
import mfem.ser as mfem
import ctypes

class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_up1 = nn.Linear(2, 16)
        self.node_up2 = nn.Linear(16, 16)

        self.gcn1 = GCNConv(16, 16)
        self.gcn2 = GCNConv(16, 16)
        self.gcn3 = GCNConv(16, 16)
        self.gcn4 = GCNConv(16, 16)
        self.gcn5 = GCNConv(16, 16)

        self.pool1 = TopKPooling(16) 

        self.node_down = nn.Linear(16, 1)

    def forward(self, data):
        x = F.gelu(self.node_up1(data.x))
        x = F.gelu(self.node_up2(x))

        x = F.gelu(self.gcn1(x, data.edge_index, data.edge_attr)) + x
        x, edge_index, edge_attr, _, _, _ = self.pool1(x, data.edge_index, data.edge_attr) 
        x = F.gelu(self.gcn2(x, edge_index, edge_attr))
        x = F.gelu(self.gcn3(x, edge_index, edge_attr)) + x
        x = F.gelu(self.gcn4(x, edge_index, edge_attr)) + x
        x = F.gelu(self.gcn5(x, edge_index, edge_attr)) + x
        x = self.node_down(x)
        return x

def main():
    filename = 'datagen/data.hdf5'

    transform = torch_geometric.transforms.Compose([
        torch_geometric.transforms.ToUndirected(),
        torch_geometric.transforms.RemoveDuplicatedEdges(),
        torch_geometric.transforms.RandomRotate(90), # rotations only affect local cartesian.
        torch_geometric.transforms.Distance(),
        torch_geometric.transforms.LocalCartesian(),
        torch_geometric.transforms.ToDevice('cpu')
    ])
    dataset = HDF5Dataset(filename, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    #model = SimpleGNN().float()
    model = GAT(2, 32, 4, 1, act='gelu', dropout=0.2, edge_dim=4, fill_value='mean').float()
    #model = GCN(2, 32, 6, 1).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    print('dataset size: ', len(dataset))

    print(train_dataset[0].edge_attr.size())

    model.train()
    for epoch in range(40):
        for batch in train_loader:
            batch.x = batch.x.float()
            batch.y = batch.y.float()
            batch.edge_attr = batch.edge_attr.float()
            pred = model(batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
            optimizer.zero_grad()
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            print(loss)

    model.eval()
    for batch in test_loader:
        batch.x = batch.x.float()
        batch.y = batch.y.float()
        batch.edge_attr = batch.edge_attr.float()
        with torch.no_grad():
            pred = model(batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
        loss = F.mse_loss(pred, batch.y)
        print('val loss: ', loss)

    data = dataset[0]
    data.x = data.x.float()
    data.edge_attr = data.edge_attr.float()
    pred = model(data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

    mesh = mfem.Mesh('/pub/afeeney/neural-schwarz-data/box3d-mesh-sol/box3d_000_refined.mesh')
    gf = mfem.GridFunction(mesh, '/pub/afeeney/neural-schwarz-data/box3d-mesh-sol/box3d_000_sol.gf')

    header = """FiniteElementSpace
FiniteElementCollection: H1_3D_P1
VDim: 1
Ordering: 0
"""

    print(gf.Size())
    print(data.y.size())

    #v = mfem.Vector()
    #gf.GetTrueDofs(v)
    with open('pred.gf', 'w+') as f:   
        f.write(header)
        f.write('\n')
        for i in range(gf.Size()):
            f.write(str(pred[i, 0].item()) + '\n')
    

'''
    header = """
FiniteElementSpace
FiniteElementCollection: H1_3D_P1
VDim: 1
Ordering: 0
"""

    with open('pred.gf', 'w+') as f:
        f.write(header)
        f.write('\n')
        for i in range(pred.size(0)):
            f.write(str(data.y[i,0].item()) + '\n')
'''

if __name__ == '__main__':
    main()
