import re
from typing import List, Tuple

import h5py
import numpy as np
import torch

from .constants import (
    normalize_temperature,
    normalize_velx,
    normalize_vely,
    normalize_dfun
)


def normalize(data, mean, abs_max):
    return (data - mean) / (abs_max - mean)


class BubbleMLForecastDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path, 
        num_input_timesteps, 
        input_step_size, 
        lead_time,
        use_mask=False
    ):
        r"""
        Args:
            data_path: path to hdf5 file
            num_input_timesteps: the history size input to the model
            input_step_size: factor of timesteps to skip.
            lead_time: how many timesteps in the future to predict.
        """
        super().__init__()
        # The simulation's nucleation model checks if a nucleation site has
        # been covered by vapor in the last four timesteps.
        assert num_input_timesteps * input_step_size >= 4

        self.handle = h5py.File(data_path, "r")

        self.num_input_timesteps = num_input_timesteps
        self.input_step_size = input_step_size
        self.lead_time = lead_time
        
        # toggles if dfun is converted to a mask of bubble posiitions.
        self.use_mask = use_mask

        self.in_channels = 4 * self.num_input_timesteps
        self.out_channels = 4 * self.lead_time

        timesteps = 700

        max_input_timestep = timesteps - lead_time - 1
        min_input_timestep = num_input_timesteps * input_step_size
        
        # the dfun for these seems very off...
        skip = ['Twall-92-49', 'Twall-93-62', 'Twall-94-81']
        self.lookup = [
            (grp_name, timestep)
            for grp_name in self.handle.keys()
            for timestep in range(min_input_timestep, max_input_timestep)
            if grp_name not in skip and self.handle[grp_name].attrs['heater_temp'] < 95
        ]

    def __del__(self):
        self.handle.close()

    def get_data(self, 
                 grp_name, 
                 start_time, 
                 end_time,
                 step,
                 use_heater_temp):
        grp = self.handle[grp_name]
        velx = normalize_velx(grp["velx"][start_time:end_time:step])
        vely = normalize_vely(grp["vely"][start_time:end_time:step])
        temperature = normalize_temperature(grp["temperature"][start_time:end_time:step])
        dfun = normalize_dfun(grp["dfun"][start_time:end_time:step])

        variables = [velx, vely, temperature, dfun]
        if use_heater_temp:
            # heater temp ranges from 90-100, so just applying a simple normalization
            heater_temp_field = (
                np.full_like(temperature, grp.attrs["heater_temp"]) - 95
            ) / 10
            variables.append(heater_temp_field)

        return np.concatenate(variables, axis=0)

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        r"""
        Each item has
            1. the xy-velocities for the liquid and vapor phases.
            2. the temperaure of the liquid and vapor phases.
            3. the signed-distance function from interface between the liquid-vapor phases.
            4. the heater temperature.
            5. the sequence of x-coordinates for nucleation sites.
        There are other variables available that are not being used: nrm_x, nrm_y, and the pressure.
        """
        grp_name, timestep = self.lookup[idx]

        # input uses a range of timesteps [start_time, t]
        start_time = timestep - self.input_step_size * self.num_input_timesteps
        target_time = timestep + self.lead_time
        input_data = self.get_data(
            grp_name, start_time, timestep, step=1, use_heater_temp=False
        )
        
        # output uses range (t, t + self.lead_time]
        output_data = self.get_data(
            grp_name, timestep + 1, target_time + 1, step=1, use_heater_temp=False
        )
        nucleation_sites_x = self.handle[grp_name]["nucleation_sites_x"][:]

        # zero pad the sequence of nucleation sites.
        # This could also use torch.nested.NestedTesnor, but this seems easier.
        # the extra zeros should not influence calculations due to the bubbleml_encoder,
        # which reduces over the sequence dim.
        pad_nuc = np.zeros(150)
        pad_nuc[:nucleation_sites_x.shape[0]] = nucleation_sites_x
        pad_nuc = np.expand_dims(pad_nuc, axis=-1)
        
        return (
            input_data.astype(np.float32),
            pad_nuc.astype(np.float32),
            output_data.astype(np.float32),
        )