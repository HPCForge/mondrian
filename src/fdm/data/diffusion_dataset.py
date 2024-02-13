import h5py
import torch
from torch.utils.data import Dataset

DIFFUSIVITY = 'diffusivity'
SOLUTION = 'solution'
XLIM = 'xlim'
YLIM = 'ylim'

class DiffusionDataset(Dataset):
    def __init__(self, filename):
        self.f = h5py.File(filename, 'r')
        self.keys = list(self.f.keys())
        # 3 timesteps + diffusivity
        self.in_channels = 4
        # remove 4 timesteps
        self.out_channels = self.f[self.keys[0]][SOLUTION].shape[0] - 4 

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        grp = self.f[key]
        diffusivity = torch.from_numpy(grp[DIFFUSIVITY][:]).unsqueeze(0)
        solution = torch.from_numpy(grp[SOLUTION][:, 1:-1, 1:-1])
        xlim = grp.attrs[XLIM]
        ylim = grp.attrs[YLIM]
        print(diffusivity.size(), solution[1:4].size())
        input = torch.cat((solution[1:4], diffusivity))
        label = solution[4:]
        print(xlim, ylim)
        return input, label, xlim, ylim

