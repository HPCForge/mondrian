import torch
import torch.nn.functional as F
from   torch import nn
from   torch.utils.data import DataLoader, random_split

from mondrian_lib.data.diffusion_dataset import diffusion_collate_fn, DiffusionDataset
from mondrian_lib.data.allen_cahn_dataset import AllenCahnDataset
from mondrian_lib.data.data_loaders import get_data_loaders
from mondrian_lib.fdm.dd_fno_mod import DDFNO
from mondrian_lib.fdm.deq_dd_fno import DDFNO as DDFNO_Modified
from mondrian_lib.fdm.deq import DEQ_DDFNO

from neuralop.losses import LpLoss
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import random

debug = False

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
    
def get_coords(xlim, ylim, input):
    x = torch.linspace(-xlim[0].item(), xlim[0].item(), input.size(-1)) / 8
    y = torch.linspace(-ylim[0].item(), ylim[0].item(), input.size(-2)) / 8
    x, y = torch.meshgrid(x, y, indexing="xy")
    x = x.unsqueeze(0).unsqueeze(0).repeat(input.size(0), 1, 1, 1)
    y = y.unsqueeze(0).unsqueeze(0).repeat(input.size(0), 1, 1, 1)
    return x, y
    
def main():
    print("Begin Training Script.")
    filename = 'datagen/fdm/allen_cahn.hdf5'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    num_epochs = 75
    batch_size = 4
    dataset = AllenCahnDataset(filename)

    print("[1] Initialize Datasets")

    pretraining = 0.01
    training = 1.0 - pretraining
    train_size = int(0.7 * len(dataset))
    val_size   = len(dataset) - train_size

    train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size, diffusion_collate_fn)

    print("[2] Initialize Models")

    premodel = DDFNO(dataset.in_channels,
                              dataset.out_channels,
                              (24,24)).float().to(device)
    
    model = DEQ_DDFNO(dataset.in_channels,
                      dataset.out_channels,
                      80,
                      (24,24)).float().to(device)

    preoptimizer = torch.optim.AdamW(premodel.parameters(), lr=0.002, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=1.5 * num_epochs * len(train_loader))

    transform = RandomRotate()

    lp_loss = LpLoss(d=2, reduce_dims=[0,1])


    if debug:
        print("Pre-Model State Dict")
        for key, value in premodel.state_dict().items():
            print("\t", key)
    
        print("Model State Dict")
        for key, value in model.state_dict().items():
            print("\t", key)

        exit(0)

    print("[3] Pre-Training")
    for epoch in range(int(num_epochs * pretraining)):
        print(f"Epoch [{epoch}/{num_epochs * pretraining}]")

        premodel.train()

        for batch in train_loader:
            preoptimizer.zero_grad()

            for size_key, (input, label, xlim, ylim) in batch.items():

                xcoords, ycoords = get_coords(xlim, ylim, input)
                input = torch.cat((input, xcoords, ycoords), dim=1)

                input = input.float().to(device)
                label = label.float().to(device)

                pred = premodel(input, xlim[0].item(), ylim[0].item())
                _lp = lp_loss(pred.detach(), label)
                _mse = F.mse_loss(pred, label)
                _mse.backward()

                print(f"Pretrain LP Loss {size_key}: ", _lp.detach())
                print(f"Pretrain MSE Loss {size_key}: ", _mse.detach())
            
            torch.nn.utils.clip_grad_norm_(premodel.parameters(), 1)
            preoptimizer.step()
    
    print("[4] Training")
    for sc, presc in zip(model.scn, premodel.scn):
        sc.load_state_dict(presc.state_dict())
    
    for ln, preln in zip(model.ln, premodel.ln):
        ln.load_state_dict(preln.state_dict())
    
    model.lifting.load_state_dict(premodel.lifting.state_dict())
    model.projection.load_state_dict(premodel.projection.state_dict())

    for epoch in range(int(num_epochs * training)):
        print(f"Epoch [{epoch}/{num_epochs * training}]")

        model.train()

        for batch in train_loader:
            optimizer.zero_grad()

            for size_key, (input, label, xlim, ylim) in batch.items():

                xcoords, ycoords = get_coords(xlim, ylim, input)
                input = torch.cat((input, xcoords, ycoords), dim=1)

                input = input.float().to(device)
                label = label.float().to(device)

                pred = model(input, xlim[0].item(), ylim[0].item())
                _lp = lp_loss(pred.detach(), label)
                _mse = F.mse_loss(pred, label)
                _mse.backward()

                print(f"Train LP Loss {size_key}: ", _lp.detach())
                print(f"Train MSE Loss {size_key}: ", _mse.detach())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    print("[5] Evaluating")

if __name__ == '__main__':
    main()