import h5py
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

EDGES = 'edges'
EDGE_LEN = 'edge-len'
VERTICES = 'vertices'
BDR_VERTEX_IDS = 'boundary-vertex-ids'
BDR_VALUES = 'boundary-values'
SOLUTION = 'solution'

class HDF5Dataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        super().__init__(None, transform=transform, pre_transform=pre_transform)
        self.file = h5py.File(filename, 'r')
        self.zfill = max(len(key) for key in self.file.keys())

    def len(self):
        return len(self.file.keys())

    def get(self, idx):
        grp = self.file[str(idx).zfill(self.zfill)]

        bdr_values = torch.tensor(grp[BDR_VALUES][:])
        bdr_ids = torch.tensor(grp[BDR_VERTEX_IDS][:]).unsqueeze(1)

        bdr_mask = torch.zeros_like(bdr_values)
        bdr_mask[bdr_ids] = 1

        x = torch.cat((bdr_values, bdr_mask), dim=1)

        return Data(x=x,
                    edge_index=torch.tensor(grp[EDGES][:]),
                    pos=torch.tensor(grp[VERTICES][:]),
                    #edge_attr=torch.tensor(grp[EDGE_LEN][:]),
                    y=torch.tensor(grp[SOLUTION][:]))
        
