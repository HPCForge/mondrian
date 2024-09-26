import h5py
import torch
from torch.utils.data import Dataset, DataLoader 

def poisson_dataloaders(data_path, batch_size):
    train_dataset = ShearLayerDataset(data_path, which="training", s=128)
    val_dataset = ShearLayerDataset(data_path, which="validation", s=128)
    test_dataset = ShearLayerDataset(data_path, which="test", s=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

class PoissonDataset(Dataset):
    def __init__(self, data, which="training", nf=0, training_samples = 1024, s=64, in_dist = True):
        
        self.in_channels = 1
        self.out_channels = 1
        
        # Note: Normalization constants for both ID and OOD should be used from the training set!
        #Load normalization constants from the TRAINING set:
        file_data_train = f"{data}/PoissonData_64x64_IN.h5"
        self.reader = h5py.File(file_data_train, 'r')
        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
        
        #The file:
        if in_dist:
            self.file_data = f"{data}/PoissonData_64x64_IN.h5"
        else:
            self.file_data = f"{data}/PoissonData_64x64_OUT.h5"

        self.s = s #Sampling rate

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024+128
            else:
                self.length = 256
                self.start = 0 
        
        #Load different resolutions
        if s!=64:
            self.file_data = "data/PoissonData_NEW_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed.
        self.reader = h5py.File(self.file_data, 'r')
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


