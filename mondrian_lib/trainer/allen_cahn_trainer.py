import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from data.diffusion_dataset import diffusion_collate_fn, DiffusionDataset
from data.allen_cahn_dataset import AllenCahnDataset
from data.kuramoto_sivashinsky_dataset import KuramotoSivashinskyDataset
from data.ns_dataset import ShearLayerDataset
from data.bubbleml_dataset import BubbleMLDataset
from data.data_loaders import get_data_loaders 
from mondrian_lib.models.fdm.dd_fno import DDFNO
import numpy as np
from neuralop.losses import LpLoss
import matplotlib
import matplotlib.pyplot as plt

import random
import torchvision.transforms as T

def _rot90(input, label, xlim, ylim):
    input = torch.rot90(input.clone(), dims=[-2, -1])
    label = torch.rot90(label.clone(), dims=[-2, -1])
    xlim, ylim = ylim, xlim
    return input, label, xlim, ylim

def _rot180(input, label, xlim, ylim):
    dims = [-2, -1]
    input = torch.rot90(input.clone(), k=2, dims=dims)
    label = torch.rot90(label.clone(), k=2, dims=dims)
    return input, label, xlim, ylim

def _rot270(input, label, xlim, ylim):
    dims = [-2, -1]
    input = torch.rot90(input.clone(), k=3, dims=dims)
    label = torch.rot90(label.clone(), k=3, dims=dims)
    return input, label, xlim, ylim

class RandomRotate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rot = [
            lambda *args: args,
            _rot90,
            _rot180,
            _rot270
        ]

    def __call__(self, input, label, xlim, ylim):
        rot_idx = torch.randint(0, len(self.rot), size=(1,)).item()
        rot_func = self.rot[rot_idx]
        return rot_func(input, label, xlim, ylim)
        
def main():
    print('started')
    filename = 'datagen/fdm/allen_cahn_3000.hdf5'
    #filename = 'datagen/fdm/kuramoto_sivashinsky.hdf5'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    num_epochs = 75
    batch_size = 8
    dataset = AllenCahnDataset(filename)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_loader, val_loader, test_loader = \
            get_data_loaders(dataset, batch_size, diffusion_collate_fn)

    model = DDFNO(
        dataset.in_channels, 
        dataset.out_channels,
        (24, 24)).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters = 1.5 * num_epochs * len(train_loader))

    """
    trans = RandomRotate()

    lp_loss = LpLoss(d=2, reduce_dims=[0, 1])

    for epoch in range(num_epochs):
        print(f'epoch [{epoch}/{num_epochs}]')
        model.train()
        for input, label in train_loader:
            optimizer.zero_grad()
            xlim, ylim = 1, 1
            input = input.float().to(device)
            label = label.float().to(device)
            pred = model(input, xlim, ylim)
            loss = lp_loss(pred.detach(), label)
            mse_loss = F.mse_loss(pred, label)
            mse_loss.backward()
            print(f'train lp loss: ', loss.detach())
            print(f'train mse: ', mse_loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for input, label in val_loader:
                xlim, ylim = 1, 1
                input = input.float().to(device)
                label = label.float().to(device)
                pred = model(input, xlim, ylim)
                optimizer.zero_grad()
                loss = F.mse_loss(pred, label).detach()
                print('val mse: ', loss)

                torch.save(input.detach().cpu(), f'4_4_input.pt')
                torch.save(pred.detach().cpu(), f'4_4_pred.pt')
                torch.save(label.detach().cpu(), f'4_4_label.pt')

    """
    trans = RandomRotate()

    lp_loss = LpLoss(d=2, reduce_dims=[0, 1])

    for epoch in range(num_epochs):
        print(f'epoch [{epoch}/{num_epochs}]')
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            for size_key, (input, label, xlim, ylim) in batch.items():
                input, label, xlim, ylim = trans(input, label, xlim, ylim)

                xcoords = torch.linspace(-xlim[0].item(), xlim[0].item(), input.size(-1)) / 8
                ycoords = torch.linspace(-ylim[0].item(), ylim[0].item(), input.size(-2)) / 8
                xcoords, ycoords = torch.meshgrid(xcoords, ycoords, indexing='xy')

                xcoords = xcoords.unsqueeze(0).unsqueeze(0).repeat(input.size(0), 1, 1, 1)
                ycoords = ycoords.unsqueeze(0).unsqueeze(0).repeat(input.size(0), 1, 1, 1)
                input = torch.cat((input, xcoords, ycoords), dim=1)

                input = input.float().to(device)
                label = label.float().to(device)
                pred = model(input, xlim[0].item(), ylim[0].item())
                loss = lp_loss(pred.detach(), label)
                #loss.backward()
                mse_loss = F.mse_loss(pred, label)
                mse_loss.backward()
                print(f'train lp loss {size_key}: ', loss.detach())
                print(f'train mse {size_key}: ', mse_loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            #lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                for size_key, (input, label, xlim, ylim) in batch.items():
                    xcoords = torch.linspace(-xlim[0].item(), xlim[0].item(), input.size(-1)) / 8
                    ycoords = torch.linspace(-ylim[0].item(), ylim[0].item(), input.size(-2)) / 8
                    xcoords, ycoords = torch.meshgrid(xcoords, ycoords, indexing='xy')

                    xcoords = xcoords.unsqueeze(0).unsqueeze(0).repeat(input.size(0), 1, 1, 1)
                    ycoords = ycoords.unsqueeze(0).unsqueeze(0).repeat(input.size(0), 1, 1, 1)
                    input = torch.cat((input, xcoords, ycoords), dim=1)

                    input = input.float().to(device)
                    label = label.float().to(device)
                    pred = model(input, xlim[0].item(), ylim[0].item())
                    optimizer.zero_grad()
                    loss = F.mse_loss(pred, label).detach()
                    print('val mse: ', loss)

                    torch.save(input.detach().cpu(), f'{size_key}_input.pt')
                    torch.save(pred.detach().cpu(), f'{size_key}_pred.pt')
                    torch.save(label.detach().cpu(), f'{size_key}_label.pt')

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
