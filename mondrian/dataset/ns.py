import h5py
import numpy as np
import torch

#DECAYING_STD = 4.22087540683
#DECAYING_STD = 6.1542138154899675
DECAYING_STD = 9.7

FIELD = 'ns_decaying_turbulence'

class PDEArenaNSDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, transform, step_range=2):
        self.transform = transform
        self.data = h5py.File(path_to_data, 'r')
        
        self.range = step_range
        self.in_channels = self.range
        self.out_channels = self.range
        
        start_time = self.range
        end_time = self.data[FIELD].shape[1] - self.range
        
        self.lookup = [
            (sim_idx, timestep)
            for sim_idx in range(self.data[FIELD].shape[0])
            for timestep in range(start_time, end_time)
        ]
    
    def __del__(self):
        self.data.close()
    
    def __len__(self):
        return len(self.lookup) # self.data['ns_decaying_turbulence'].shape[0]
    
    def __getitem__(self, idx):       
        sim_idx, timestep = self.lookup[idx]        
        input = (self.data[FIELD][sim_idx])[timestep - self.range:timestep] / DECAYING_STD
        label = (self.data[FIELD][sim_idx])[timestep:timestep + self.range] / DECAYING_STD

        if self.transform:
            #num_noise_levels = 10
            #noise_levels = np.linspace(0, 1, num_noise_levels)
            #l = np.random.randint(0, num_noise_levels)
            #input += np.random.standard_normal(input.shape) * noise_levels[l]

            k = np.random.randint(0,4)
            input = np.ascontiguousarray(np.rot90(input, k=k, axes=(-2, -1)))
            label = np.ascontiguousarray(np.rot90(label, k=k, axes=(-2, -1)))
        
        return input, label