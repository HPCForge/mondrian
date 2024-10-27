import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def imshow(ax, img, vmin, vmax, title):
    im = ax.imshow(torch.rot90(img),
              cmap='jet',
              vmin=vmin,
              vmax=vmax)
    ax.title.set_text(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, required=True)
args = parser.parse_args()

save_dir = Path(f'{args.log_dir}/')
save_dir.mkdir(parents=True, exist_ok=True)
print(f'saving to {save_dir}')

input = torch.load(f'{args.log_dir}/Input.pt')
label = torch.load(f'{args.log_dir}/Label.pt').detach().cpu()
pred = torch.load(f'{args.log_dir}/Pred.pt').detach().cpu()

vmin, vmax = input.min(), input.max()

for idx in range(input.size(0)):
    fig, axarr = plt.subplots(1, 3, figsize=(5, 12))
    im1 = imshow(axarr[0], input[idx, 0], vmin, vmax, 'Input (T=0)')
    imshow(axarr[1], label[idx, 0], vmin, vmax, 'Ground Truth (T=1)')
    imshow(axarr[2], pred[idx, 0], vmin, vmax, 'Predicted (T=1)')
    fig.colorbar(im1, ax=axarr, fraction=0.026, pad=0.04)
    plt.savefig(f'{save_dir}/{idx}.png')
    plt.close()
