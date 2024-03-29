import h5py
import torch
from torch.utils.data import Dataset, default_collate

NU = 'nu'
ALPHA = 'alpha'
SOLUTION = 'solution'
FIELD_KEYS = [
    SOLUTION
]
ATTR_KEYS = [
    NU, ALPHA
]

XLIM = 'xlim'
YLIM = 'ylim'

class KuramotoSivashinskyDataset(Dataset):
    def __init__(self, filename):
        self.f = h5py.File(filename, 'r')
        
        # simulations are grouped by size.
        self.keys = []
        size_group_keys = list(self.f.keys())
        for size_group_key in size_group_keys:
            for sim_key in self.f[size_group_key].keys():
                self.keys.append((size_group_key, sim_key))

        self._min, self._max = self._extents()

        # 4 timesteps + coords (added after augmentations)
        self.t_start = 8
        self.in_channels = self.t_start + 2
        size_key, sim_key = self.keys[0]
        self.out_channels = self.f[size_key][sim_key][SOLUTION].shape[0] - self.t_start 

    def _extents(self):
        _min = {NU: float('inf'), ALPHA: float('inf'), SOLUTION: float('inf')}
        _max = {NU: float('-inf'), ALPHA: float('-inf'), SOLUTION: float('-inf')}
        for size_key, sim_key in self.keys:
            for field_key in FIELD_KEYS:
                field = self.f[size_key][sim_key][field_key][:]
                _min[field_key] = min(_min[field_key], field.min())
                _max[field_key] = max(_max[field_key], field.max())
            for field_key in ATTR_KEYS:
                field = self.f[size_key][sim_key].attrs[field_key]
                _min[field_key] = min(_min[field_key], field)
                _max[field_key] = max(_max[field_key], field)
        return _min, _max

    def _normalize(self, data, field_key):
        # sort of shift mean closer to zero
        data = data - 2
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
        xlim = grp.attrs[XLIM]
        ylim = grp.attrs[YLIM]
        # TODO: add ALPHA and NU, currently they are constant!
        input = solution[:self.t_start]
        label = solution[self.t_start:]
        return size_key, input, label, xlim, ylim
