debug = True

def train(dset, filename):

    from torch.utils.data import DataLoader
    from torchdeq.core import DEQBase
    import matplotlib.pyplot as plt
    from mondrian_lib.fdm.torchdeq_dd_fno import DEQ_DDFNO
    from mondrian_lib.data.data_loaders import get_data_loaders
    from mondrian_lib.data.diffusion_dataset import diffusion_collate_fn
    import torch
    import torch.nn.functional as F
    import numpy as np

    print("Begin Training Script")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    dataset = dset(filename)

    print("Initializing Datasets")
    train_loader, val_loader, test_loader = \
            get_data_loaders(dataset, batch_size, diffusion_collate_fn)

    print("Initializing Model")


    model = DEQ_DDFNO(dataset.in_channels, 
                      dataset.out_channels, 
                      (24, 24))
    model = model.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    print("Begin Training")
    for epoch in range(40):
        model.train()
        for index, batch in enumerate(train_loader):
            print("Batch Index: ", index)
            optimizer.zero_grad()
            for size_key, (input, label, xlim, ylim) in batch.items():
                #assert(not np.any(np.isnan(x)) for x in input)
                if debug: 
                    print("\tEpoch: ", epoch, " : ", end="")
                    print("Input -> ", end="")
                input = input.float()
                label = label.float()
                input = input.to(device)
                label = label.to(device)

                if debug: print("DEQ -> ", end="")
                z_out = model(input, xlim[0].item(), ylim[0].item())

                if debug: print("Loss -> ", end="")
                loss = F.mse_loss(z_out, label)
                loss.backward()
                print(f"Train MSE {size_key}: ", loss.detach())
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

if __name__ == '__main__':
    mode = 1

    if mode == 0:
        from mondrian_lib.data.diffusion_dataset import DiffusionDataset
        train(DiffusionDataset, 'datagen/fdm/diffusion.hdf5')
    else:
        from mondrian_lib.data.allen_cahn_dataset import AllenCahnDataset
        train(AllenCahnDataset, 'datagen/fdm/allen_cahn.hdf5')
