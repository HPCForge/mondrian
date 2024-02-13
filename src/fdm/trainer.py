import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.diffusion_dataset import DiffusionDataset
from mondrian_lib.models.fdm.dd_fno import DDFNO
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
    filename = 'datagen/fdm/diffusion.hdf5'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 2 

    dataset = DiffusionDataset(filename)

    train_size = int(0.7 * len(dataset))
    test_and_val_size = len(dataset) - train_size
    val_size = int(0.5 * test_and_val_size)
    test_size = test_and_val_size - val_size

    train_dataset, val_dataset, test_dataset = \
            torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DDFNO(
        dataset.in_channels, 
        dataset.out_channels,
        (16, 16)).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(10):
        print(model)
        model.train()
        for input, label, xlim, ylim in train_loader:
            input = input.to(float).to(device)
            label = label.to(float).to(device)
            pred = model(input, xlim, ylim)
            optimizer.zero_grad()
            loss = F.mse_loss(pred, label)
            loss.backward()
            optimizer.step()
            print('train mse: ', loss)

        #model.eval()
        #for batch in val_loader:
        #    with torch.no_grad():
        #        batch = batch.to(device)
        #        pred = model(batch)
        #        loss = F.mse_loss(pred, batch.y)
        #    print('val mse: ', loss)

    #model.eval()
    #accum_test_loss = 0
    #for batch in test_loader:
    #    with torch.no_grad():
    #        batch = batch.to(device)
    #        pred = model(batch)
    #        loss = F.mse_loss(pred, batch.y, reduction='mean')
    #        print('test mse: ', loss)
    #        accum_test_loss += F.mse_loss(pred, batch.y, reduction='sum')
    #test_mse = accum_test_loss / (len(test_loader) * batch_size)
    #print('total test mse: ', test_mse) 

if __name__ == '__main__':
    main()
