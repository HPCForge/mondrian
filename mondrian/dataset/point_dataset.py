import einops
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

class PointDataset(Dataset):
    r"""
    Converts an existing dataset to its point-wise evaluation
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.in_channels = self.dataset.in_channels
        self.out_channels = self.dataset.out_channels

    def __len__(self):
        return len(self.dataset)

    def _coords(self, x_res, y_res):
        x_coords = torch.linspace(-1, 1, x_res + 2)[1:-1]
        y_coords = torch.linspace(-1, 1, y_res + 2)[1:-1]
        coords = torch.stack(torch.meshgrid(x_coords, y_coords, indexing='xy'),
                             dim=-1)
        return einops.rearrange(coords, 'H W d -> (H W) d')

    def __getitem__(self, idx):
        # [c, H, W]
        input, label = self.dataset[idx]

        x_res, y_res = input.size(2), input.size(1)
        c = self._coords(x_res, y_res)

        input = einops.rearrange(input, 'c H W -> (H W) c')
        label = einops.rearrange(label, 'c H W -> (H W) c')

        assert input.size(0) == c.size(0)
        assert label.size(0) == c.size(0)

        data = Data(x=input,
                    y=label,
                    pos=c)
                    #src_coords=c,
                    #tgt_coords=c)

        return data