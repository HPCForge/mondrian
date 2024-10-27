import h5py
import torch
from torch.utils.data import Dataset, DataLoader 

def shear_layer_dataloaders(data_path, batch_size):
    train_dataset = ShearLayerDataset(data_path, which="training", s=128)
    val_dataset = ShearLayerDataset(data_path, which="validation", s=128)
    test_dataset = ShearLayerDataset(data_path, which="test", s=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

class ShearLayerDataset(Dataset):
    def __init__(self, data, which="training", nf=0, training_samples = 750, s=64, in_dist = True):

        self.in_channels = 1
        self.out_channels = 1
        self.s = s
        self.in_dist = in_dist

        if in_dist:
            if self.s==64:
                self.file_data = f"{data}/NavierStokes_64x64_IN.h5" #In-distribution file 64x64
            else:
                self.file_data = f"{data}/NavierStokes_128x128_IN.h5"   #In-distribution file 128x128
        else:
            self.file_data = f"{data}/NavierStokes_128x128_OUT.h5"  #Out-of_-distribution file 128x128

        self.reader = h5py.File(self.file_data, 'r')
        self.N_max = 1024

        self.n_val  = 128
        self.n_test = 128
        self.min_data = 1.4307903051376343
        self.max_data = -1.4307903051376343
        self.min_model = 2.0603253841400146
        self.max_model= -2.0383243560791016

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = self.n_val
            self.start = self.N_max - self.n_val - self.n_test
        elif which == "test":
            self.length = self.n_test
            self.start = self.N_max  - self.n_test

        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        if self.s == 64 and self.in_dist:
            inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
            labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        else:

            inputs = self.reader['Sample_' + str(index + self.start)]["input"][:].reshape(1,1,self.s, self.s)
            labels = self.reader['Sample_' + str(index + self.start)]["output"][:].reshape(1,1, self.s, self.s)

            if self.s<128:
                inputs = downsample(inputs, self.s).reshape(1, self.s, self.s)
                labels = downsample(labels, self.s).reshape(1, self.s, self.s)
            else:
                inputs = inputs.reshape(1, 128, 128)
                labels = labels.reshape(1, 128, 128)

            inputs = torch.from_numpy(inputs).type(torch.float32)
            labels = torch.from_numpy(labels).type(torch.float32)

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

