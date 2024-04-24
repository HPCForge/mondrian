import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
import h5py
import random
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF
from pathlib import Path

TEMPERATURE = 'temperature'
PRESSURE = 'pressure'
X_VELOCITY = 'velx'
Y_VELOCITY = 'vely'
DISTANCE_FUNC = 'dfun'
MASS_FLUX = 'mass_flux'

NUCLEATION_SITES_X = 'nucleation_sites_x'

FIELDS = [
    TEMPERATURE,
    #PRESSURE,
    X_VELOCITY,
    Y_VELOCITY,
    #DISTANCE_FUNC,
    MASS_FLUX
]

# The max_nuc_sites assumes the linear dependence.
# and a max temperature of 110
MAX_NUC_SITES = 110 - 50

NUC_SITE_EMBED_SIZE = 32

class BubbleMLDataset(Dataset):
    def __init__(self,
                 filename,
                 style,
                 dtype=torch.float32,
                 history_window=10,
                 future_window=10):
        r"""
        Args:
            filename: path to the hdf5 file.
            style: 'train' or 'test'. A training dataset considers any overlapping
                   slices, while the test set uses non-overlapping slices
            dtype: The input/label tensor's datatype.
            history_window: The number of timesteps passed as input to the model
            future_window: The number of timesteps the model predicts
        """
        super().__init__()
        assert style in ('train', 'test')

        # nucleation model basically creates a bubble if
        # the cell has been uncovered by vapor for 4
        # timesteps. So, we need to have at least four
        # prior timesteps to know if a bubble may form 
        assert history_window >= 4

        self.filename = filename
        self.dtype=dtype
        self.history_window = history_window
        self.future_window = future_window

        self.file = h5py.File(self.filename, 'r')
        self.sim_keys = list(self.file.keys())

        self.t_max = self.file[self.sim_keys[0]][TEMPERATURE].shape[0]

        time_lim = self.t_max - history_window - future_window

        if style == 'train':
            # a set of unique indices for all the simulations and timesteps.
            # Simulations go from [0, T_max]. We only consider
            # [0, T_max - history_window - future_window] for the input range
            valid_timesteps = range(0, time_lim, 10)
        else:
            # Only consider non-overlapping slices from each simulation
            valid_timesteps = range(0, time_lim, 2 * (history_window + future_window))
        self.indices = [
            (sim_id, timestep) for sim_id in self.sim_keys
                               for timestep in valid_timesteps
        ]

        # There are five fields corresponding to temperature,
        # pressure, x-velocity, y-velocity, and the distance
        # function from bubble interfaces
        # TODO: add something for the nucleation sites
        self.num_input_fields = len(FIELDS)
        self.in_channels = self.num_input_fields * self.history_window + NUC_SITE_EMBED_SIZE 

        # We use the same five output fields.
        self.num_output_fields = len(FIELDS)
        self.out_channels = self.num_output_fields * self.future_window

    def __len__(self):
        return len(self.indices)

    def history_range(self, timestep):
        return timestep, timestep + self.history_window

    def future_range(self, timestep):
        history = timestep + self.history_window
        return history, history + self.future_window

    def _get_history(self, sim, timestep):
        lo, hi = self.history_range(timestep)
        history = np.concatenate(
            [sim[field][lo:hi] for field in FIELDS], axis=0)
        return torch.from_numpy(history).to(self.dtype)
    
    def _get_future(self, sim, timestep):
        lo, hi = self.future_range(timestep)
        future = np.concatenate(
            [sim[field][lo:hi] for field in FIELDS], axis=0)
        return torch.from_numpy(future).to(self.dtype)

    def __getitem__(self, idx):
        sim_id, timestep = self.indices[idx]
        sim = self.file[sim_id]

        history = self._get_history(sim, timestep)
        future = self._get_future(sim, timestep)

        # map nuc_sites in [-2.5, 2.5] to a range of indices [1, 501]
        nuc_sites = torch.from_numpy(sim[NUCLEATION_SITES_X][:])
        assert nuc_sites.min() >= -2.51
        assert nuc_sites.max() <= 2.51
        nuc_indices = torch.round(torch.clamp(nuc_sites, -2.5, 2.5) * 100 + 251).to(int)

        # append 0s to reach some target length
        target_length = MAX_NUC_SITES - nuc_indices.size(0) + 1
        zeros = torch.zeros(target_length).to(int)
        nuc_indices = torch.cat((nuc_indices, zeros))

        return history, nuc_indices, future
