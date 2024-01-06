import h5py
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

COORDS = 'interior_coords'
SOLUTION = 'interior_solution'
FORCING = 'interior_forcing'
COEFF = 'interior_coeff'
BOUNDARY_COORDS = 'boundary_coords'
BOUNDARY_VALUES = 'boundary_values'

class HDF5Dataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        super().__init__(None, transform=transform, pre_transform=pre_transform)
        self._file = h5py.File(filename, 'r')
        self.zfill = max(len(key) for key in self._file.keys())
        self.min_value, self.max_value = self.extents([SOLUTION, BOUNDARY_VALUES])
        self.min_forcing, self.max_forcing = self.extents([FORCING])
        self.min_coeff, self.max_coeff = self.extents([COEFF])

    def len(self):
        r""" len is the number of hdf5 groups
        """
        return len(self._file.keys())

    def extents(self, keys):
        min_, max_ = float('inf'), -float('inf')
        for group_key in self._file.keys():
            sim_group = self._file[group_key]
            for key in keys:
                min_ = min(min_, sim_group[key][:].min())
                max_ = max(max_, sim_group[key][:].max())
        return min_, max_

    def normalize(self, x, lo, hi):
        if abs(lo - hi) < 1e-8:
            return x
        return (x - lo) / (hi - lo) * 2 - 1 

    def get(self, idx):
        sim_group = self._file[str(idx).zfill(self.zfill)]

        # forcing is not applied on boundary vertices
        boundary_coords = torch.tensor(sim_group[BOUNDARY_COORDS][:])
        boundary_values = torch.tensor(sim_group[BOUNDARY_VALUES][:])
        boundary_forcing = torch.zeros_like(boundary_values)
        boundary_coeff = torch.zeros_like(boundary_values)

        # interior values are initialized to zero
        interior_coords = torch.tensor(sim_group[COORDS][:])
        interior_forcing = torch.tensor(sim_group[FORCING][:])
        interior_coeff = torch.tensor(sim_group[COEFF][:])
        # values are unknown, so initialize with zero
        interior_values = torch.zeros_like(interior_forcing)

        # normalize to [-1, 1], based on max that appear in the entire dataset
        boundary_values = self.normalize(boundary_values, self.min_value, self.max_value)
        interior_forcing = self.normalize(interior_forcing, self.min_forcing, self.max_forcing)
        interior_coeff = self.normalize(interior_coeff, self.min_coeff, self.max_coeff)

        # construct node features
        boundary_features = torch.cat(
                (boundary_values, boundary_forcing, boundary_coeff),
                dim=1)
        interior_features = torch.cat(
                (interior_values, interior_forcing, interior_coeff),
                dim=1)
        features = torch.cat((boundary_features, interior_features), dim=0)

        boundary_indices = torch.arange(boundary_coords.size(0))

        # track which nodes are boundary, and which are not.
        boundary_mask = torch.zeros((features.size(0), 1))
        boundary_mask[boundary_indices] = 1
        boundary_mask -= 0.5
        x = torch.cat((features, boundary_mask), dim=1)

        # construct solution, boundary values and interior values
        solution = self.normalize(torch.tensor(sim_group[SOLUTION][:]), self.min_value, self.max_value)
        y = torch.cat((boundary_values, solution), dim=0)

        pos = torch.cat((boundary_coords, interior_coords), dim=0)

        return Data(x=x.float(),
                    y=y.float(),
                    pos=pos.float(),
                    boundary_indices=boundary_indices) 
