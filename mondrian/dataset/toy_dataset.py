import h5py
import torch
from torch import nn


class ToyInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, path, dtype=torch.float32):
        super().__init__()
        with h5py.File(path, "r") as handle:
            self.input = torch.from_numpy(handle["input"][:]).to(dtype)
            self.label = torch.from_numpy(handle["label"][:]).to(dtype)
        self.in_channels = 1
        self.out_channels = 1

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return (self.input[idx].unsqueeze(0), self.label[idx].unsqueeze(0))
