import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.diffusion_dataset import diffusion_collate_fn, DiffusionDataset
from data.allen_cahn_dataset import AllenCahnDataset
from data.data_loaders import get_data_loaders 
from mondrian_lib.models.fdm.dd_fno import DDFNO
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import random
import torchvision.transforms as T

class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]

transforms = RandomChoice([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
])

def main():
    print('started')
    filename = 'datagen/fdm/allen_cahn.hdf5'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    batch_size = 16 
    #dataset = DiffusionDataset(filename)
    dataset = AllenCahnDataset(filename)
    train_loader, val_loader, test_loader = \
            get_data_loaders(dataset, batch_size, diffusion_collate_fn)

    model = DDFNO(
        dataset.in_channels, 
        dataset.out_channels,
        (24, 24)).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(40):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            for size_key, (input, label, xlim, ylim) in batch.items():
                #[input, label] = transforms([input, label])
                # TODO: the vertical/horizontal flip only work if xlim == ylim
                #input, label = transforms([input, label])
                input = input.float().to(device)
                label = label.float().to(device)
                pred = model(input, xlim[0].item(), ylim[0].item())
                loss = F.mse_loss(pred, label)
                loss.backward()
                print(f'train mse {size_key}: ', loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                for size_key, (input, label, xlim, ylim) in batch.items():
                    input = input.float().to(device)
                    label = label.float().to(device)
                    pred = model(input, xlim[0].item(), ylim[0].item())
                    optimizer.zero_grad()
                    loss = F.mse_loss(pred, label).detach()
                    print('val mse: ', loss)

                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(pred[0][50].detach().cpu(), vmin=-1, vmax=1, cmap='turbo')
                    ax[1].imshow(label[0][50].detach().cpu(), vmin=-1, vmax=1, cmap='turbo')
                    plt.savefig(f'{size_key}.png')
                    plt.close()
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
