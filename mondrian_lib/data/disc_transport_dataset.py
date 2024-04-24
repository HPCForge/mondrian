import h5py
import torch
from torch.utils.data import Dataset, DataLoader 

def disc_transport_dataloaders(data_path, batch_size):
    train_dataset = DiscTransportDataset(data_path, which="training", s=128)
    val_dataset = DiscTransportDataset(data_path, which="validation", s=128)
    test_dataset = DiscTransportDataset(data_path, which="test", s=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

class DiscTransportDataset(Dataset):
    def __init__(self, data, which, nf=0, training_samples = 512, s = 64, in_dist = True):
        
        self.in_channels = 1
        self.out_channels = 1
        
        if in_dist:
            self.file_data = f"{data}/DiscTranslation_64x64_IN.h5"
        else:
            self.file_data = f"{data}/DiscTranslation_64x64_OUT.h5"
        
        if which == "training":
            self.length = training_samples
            self.start = 0
            
        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512+256
            else:
                self.length = 256
                self.start = 0

        self.reader = h5py.File(self.file_data, 'r') 

        #Default:
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)


        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid
