import h5py
import torch
from torch import nn


class AllenCahnInMemoryDataset(nn.Module):
    r"""
    This is a simple Allen Cahn dataset that is small enough to fit in CPU memory.
    This is mostly intended to be for fast testing.
    The data has two important keys:
      1. solution, which contains the initial condition and target at indices 0 and 1.
      2. The diffusivity, which is just a float.
    """

    def __init__(self, path, dtype=torch.float32):
        super().__init__()
        with h5py.File(path, "r") as handle:
            print(handle.keys())
            for res_group in handle.keys():
                grp = handle[res_group]
                self.solution = torch.stack(
                    [torch.from_numpy(grp[key]["solution"][:]) for key in grp.keys()]
                ).to(dtype)
                self.diffusivity = torch.tensor(
                    [grp[key].attrs["diffusivity"] for key in grp.keys()]
                ).to(dtype)

        self.in_channels = 2
        self.out_channels = 1

    def __len__(self):
        return len(self.diffusivity)

    def __getitem__(self, idx):
        in_sol = self.solution[idx][0]
        in_diff = torch.full_like(in_sol, self.diffusivity[idx])

        input = torch.stack((in_sol, in_diff), dim=0)
        output = self.solution[idx][1].unsqueeze(0)

        return input, output
