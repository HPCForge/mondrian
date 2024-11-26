import h5py
import torch
from torch.utils.data import Dataset, default_collate

DIFFUSIVITY = 'diffusivity'
SOLUTION = 'solution'
XLIM = 'xlim'
YLIM = 'ylim'

class AllenCahnDataset(Dataset):
    def __init__(self, filename, which='training', in_steps=1, out_steps=30):
        assert in_steps + out_steps <= 31
        self.f = h5py.File(filename, 'r')
        self.keys = []
        # diffusivity is the extra channel
        self.in_steps = in_steps
        self.in_channels = in_steps + 1
        self.out_channels = out_steps
        datasize = {'training': 750, 'validation': 50, 'test': 200}
        size_group_keys = list(self.f.keys())
        for size_group_key in size_group_keys:
            for sim_key in self.f[size_group_key].keys():
                self.keys.append((size_group_key, sim_key))

        if which == 'training':
            self.keys = self.keys[:datasize['training']]
        elif which == 'validation':
            self.keys = self.keys[datasize['training']:datasize['training']+datasize['validation']]
        elif which == 'test':
            self.keys = self.keys[-datasize['test']:]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        size_key, sim_key = self.keys[idx]
        ic = torch.from_numpy(
            self.f[size_key][sim_key][SOLUTION][:self.in_steps]
        )
        # preprocess diffusivity feature
        diff = self.f[size_key][sim_key].attrs[DIFFUSIVITY]
        diff_feature = torch.ones_like(ic[0]) * diff

        # add diffusivity to input
        input = torch.cat([ic, diff_feature.unsqueeze(0)], dim=0).float()
        label = torch.from_numpy(
            self.f[size_key][sim_key][SOLUTION][self.in_steps:self.in_steps + self.out_channels]
        ).float()
        return input, label
