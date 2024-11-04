import h5py
import torch
from torch.utils.data import Dataset, default_collate
from imgs.viz import viz_data
DIFFUSIVITY = 'diffusivity'
SOLUTION = 'solution'
XLIM = 'xlim'
YLIM = 'ylim'

class AllenCahnDataset(Dataset):
    def __init__(self, filename, which='training'):

        assert which in ('training', 'validation', 'test')

        if which == 'training':
            self.f = h5py.File(f"{filename}/train.hdf5", 'r')
        elif which == 'validation':
            self.f = h5py.File(f"{filename}/valid.hdf5", 'r')
        elif which == 'test':
            self.f = h5py.File(f"{filename}/test.hdf5", 'r')    
        # simulations are grouped by size.
        self.keys = []
        size_group_keys = list(self.f.keys())
        for size_group_key in size_group_keys:
            for sim_key in self.f[size_group_key]:
                self.keys.append(sim_key)
                
        self.in_channels = 1
        self.out_channels = 30

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        input = self.keys[idx][:self.in_channels]
        label = self.keys[idx][self.in_channels:]
        return input, label


if __name__ == '__main__':
    dataset = AllenCahnDataset('/share/crsp/lab/ai4ts/share/BubbleML_f32/AllenCahn/')
    print(len(dataset))
    print(f'shape of an input: {dataset[0].shape}')
    print(f'type of input: {type(dataset[0])}')
    viz_data(dataset[0][-1])