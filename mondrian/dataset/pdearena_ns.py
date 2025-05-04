import h5py
import numpy as np
import torch

class PDEArenaNSDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, transform):
        self.transform = transform
        # train dataset is only 26GB, so can fit in CPU memory...
        with h5py.File(path_to_data) as handle:
            print(handle.keys())
            # each sim is conditioned on different y-buoyancy
            self.buo_y = handle['buo_y'][:]
            # u is a particle concentration
            self.u = handle['u'][:]
            self.vx = handle['vx'][:]
            self.vy = handle['vy'][:]
    
        self.in_range = np.arange(0, 5)
        self.out_range = np.array([5, 10, 15])
        
        self.in_channels = 3 * len(self.in_range) + 1
        self.out_channels = 3 * len(self.out_range)
    
    def __len__(self):
        return self.buo_y.shape[0] * 2
    
    def __getitem__(self, idx):
        # Simulations are 
        # Each sample is from the first or second half of the trajectory
        if idx >= self.buo_y.shape[0]:
            start_time = 26
            idx -= self.buo_y.shape[0]
        else:
            start_time = 0
        
        in_range = start_time + self.in_range
        out_range = start_time + self.out_range
        
        in_u = self.u[idx, in_range]
        in_vx = self.vx[idx, in_range]
        in_vy = self.vy[idx, in_range]
        in_buo_y = np.expand_dims(np.full_like(in_vx[0], self.buo_y[idx]), 0)
        input = np.concatenate([in_u, in_vx, in_vy, in_buo_y], axis=0)
        
        label_u = self.u[idx, out_range]
        label_vx = self.vx[idx, out_range]
        label_vy = self.vy[idx, out_range]
        label = np.concatenate([label_u, label_vx, label_vy], axis=0)
                
        if self.transform:
            input += np.random.standard_normal(input.shape) * 0.1
            k = np.random.randint(0,4)
            input = np.ascontiguousarray(np.rot90(input, k=k, axes=(-2, -1)))
            label = np.ascontiguousarray(np.rot90(label, k=k, axes=(-2, -1)))
        
        return input, label