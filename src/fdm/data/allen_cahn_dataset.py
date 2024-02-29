import h5py
import torch
from torch.utils.data import Dataset, default_collate

DIFFUSIVITY = 'diffusivity'
SOLUTION = 'solution'
XLIM = 'xlim'
YLIM = 'ylim'

class AllenCahnDataset(Dataset):
    def __init__(self, filename):
        self.f = h5py.File(filename, 'r')
        
        # simulations are grouped by size.
        self.keys = []
        size_group_keys = list(self.f.keys())
        for size_group_key in size_group_keys:
            for sim_key in self.f[size_group_key].keys():
                self.keys.append((size_group_key, sim_key))

        self._min, self._max = self._extents()

        # 6 timesteps + diffusivity + coords
        self.in_channels = 9
        # remove 4 timesteps
        size_key, sim_key = self.keys[0]
        self.out_channels = self.f[size_key][sim_key][SOLUTION].shape[0] - 7 

    def _extents(self):
        _min = {DIFFUSIVITY: float('inf'), SOLUTION: float('inf')}
        _max = {DIFFUSIVITY: float('-inf'), SOLUTION: float('-inf')}
        for size_key, sim_key in self.keys:
            diff = self.f[size_key][sim_key].attrs[DIFFUSIVITY]
            _min[DIFFUSIVITY] = min(_min[DIFFUSIVITY], diff.min())
            _max[DIFFUSIVITY] = max(_max[DIFFUSIVITY], diff.max())
            sol = self.f[size_key][sim_key][SOLUTION][:]
            _min[SOLUTION] = min(_min[SOLUTION], sol.min())
            _max[SOLUTION] = max(_max[SOLUTION], sol.max())
        return _min, _max

    def _normalize(self, data, field_key):
        _min = self._min[field_key]
        _max = self._max[field_key]
        return 2 * ((data - _min) / (_max - _min)) - 1

    def _lookup_normalized(self, grp, key):
        field = torch.from_numpy(grp[key][:])
        return self._normalize(field, key)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        size_key, sim_key = self.keys[idx]
        grp = self.f[size_key][sim_key]
        solution = self._lookup_normalized(grp, SOLUTION)
        diffusivity_scalar = self._normalize(grp.attrs[DIFFUSIVITY], DIFFUSIVITY)
        diffusivity = torch.full_like(solution[0], diffusivity_scalar).unsqueeze(0)
        xlim = grp.attrs[XLIM]
        ylim = grp.attrs[YLIM]
        xcoords = torch.linspace(-xlim, xlim, solution.size(2)) / 8
        ycoords = torch.linspace(-ylim, ylim, solution.size(1)) / 8
        xcoords, ycoords = torch.meshgrid(xcoords, ycoords, indexing='xy')
        xcoords, ycoords = xcoords.unsqueeze(0), ycoords.unsqueeze(0)
        input = torch.cat((solution[1:7], diffusivity, xcoords, ycoords))
        label = solution[7:]
        print(input.size(), label.size())
        return size_key, input, label, xlim, ylim
