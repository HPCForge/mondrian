import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader
from fdm_poisson import read_boundary, write_boundary, boundary_to_vec
import matplotlib.pyplot as plt

class Solver(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 128
        self._reset()

    def _reset(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1000),
            nn.GELU(),
            nn.Linear(1000, self.output_size)
        )

    def forward(self, x):
        return self.model(x)

class HDF5Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.file = h5py.File(path, 'r')
        self.zfill_cnt = max([len(k) for k in self.file.keys()])

    def __len__(self):
        return len(self.file.keys())

    def __getitem__(self, idx):
        idx_str = str(idx).zfill(self.zfill_cnt)
        sol = self.file[idx_str]['sol'][:]
        bc = read_boundary(sol)
        input = torch.from_numpy(boundary_to_vec(bc)).float()
        label = torch.from_numpy(sol).flatten().float()
        return input, label

def main():
    dataset = HDF5Dataset('./poisson.hdf5')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_datset = torch.utils.data.random_split(dataset, (train_size, test_size))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    sample_input, sample_label = dataset[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Solver(sample_input.size(0), sample_label.size(0)).float().to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters())

    epochs = 10
    for epoch in range(epochs):
        print(f'Epoch: [{epoch}/{epochs}]')
        for input, label in train_dataloader:
            input = input.to(device)
            label = label.to(device)
            pred = model(input)
            optimizer.zero_grad()
            loss = F.mse_loss(pred, label)
            loss.backward()
            optimizer.step()

        for input, label in val_dataloader:
            input = input.to(device)
            label = label.to(device)
            pred = model(input)
            loss = F.mse_loss(pred, label)
            print('val: ', loss)

        pred = pred.detach().reshape((-1, 100, 100))
        label = label.detach().reshape((-1, 100, 100))

        # fft in y-direction.
        pred_h_y = torch.fft.rfft(pred).abs().square().mean(dim=-2)
        label_h_y = torch.fft.rfft(label).abs().square().mean(dim=-2)

        rel_err = torch.fft.rfft(abs(pred - label)).abs().square().mean(dim=-2)

        size = pred_h_y.size(1) - 1
        fig, axarr = plt.subplots(1, 2)
        axarr[0].plot(range(size), pred_h_y[0, 1:], label='pred')
        axarr[0].plot(range(size), label_h_y[0, 1:], label='label')
        axarr[0].legend()
        axarr[0].set_yscale('log')
        axarr[1].plot(range(size), rel_err[0, 1:])
        plt.savefig('poisson_sol.png')
        plt.close()

        #fig, axes = plt.subplots(1, 2)
        #axes[0].imshow(pred_h_y.cpu()[0])
        #axes[1].imshow(label_h_y.cpu()[0])
        #plt.tight_layout()

    torch.save(model, 'model.pt')

if __name__ == '__main__':
    main()
