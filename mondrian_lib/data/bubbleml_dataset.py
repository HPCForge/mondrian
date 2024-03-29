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

FIELDS = [
    TEMPERATURE,
    PRESSURE,
    X_VELOCITY,
    Y_VELOCITY,
    DISTANCE_FUNC
]

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

        self.filename = filename
        self.dtype=dtype
        self.history_window = history_window
        self.future_window = future_window

        self.in_channels = len(FIELDS) * history_window
        self.out_channels = len(FIELDS) * future_window

        self.file = h5py.File(self.filename, 'r')
        self.sim_keys = list(self.file.keys())

        self.t_max = self.file[self.sim_keys[0]][TEMPERATURE].shape[0]

        time_lim = self.t_max - history_window - future_window

        if style == 'train':
            # a set of unique indices for all the simulations and timesteps.
            # Simulations go from [0, T_max]. We only consider
            # [0, T_max - history_window - future_window] for the input range
            valid_timesteps = range(time_lim)
        else:
            # Only consider non-overlapping slices from each simulation
            valid_timesteps = range(0, time_lim, history_window + future_window)
        self.indices = [
            (sim_id, timestep) for sim_id in self.sim_keys
                               for timestep in valid_timesteps
        ]

        # There are five fields corresponding to temperature,
        # pressure, x-velocity, y-velocity, and the distance
        # function from bubble interfaces
        # TODO: add something for the nucleation sites
        self.num_input_fields = len(FIELDS)
        self.in_channels = self.num_input_fields * self.history_window 

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

        return history, future
