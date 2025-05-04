from omegaconf import OmegaConf
import hydra
import pathlib
import functools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as ani
#import cmocean
#from cmap import Colormap
import seaborn

import torch
from torch.utils.data import DataLoader
import lightning as L

from mondrian.trainer.simple_trainer import SimpleModule
from mondrian.dataset.ns import PDEArenaNSDataset

from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

font = {'size' : 15}

matplotlib.rc('font', **font)


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")

    module = SimpleModule.load_from_checkpoint(cfg.model_ckpt_path)
    module = module.cpu()
    
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method(cfg.experiment.quadrature_method)
    set_default_qkv_operator(cfg.experiment.linear_operator)
    set_default_feed_forward_operator(cfg.experiment.neural_operator)

    test_datasets = [
        PDEArenaNSDataset(data_path, False)
        for data_path in [cfg.experiment.test_data_path]
    ]
    
    batch = test_datasets[0][2]
    input32, label32 = batch
    input32 = torch.from_numpy(input32)
    label32 = torch.from_numpy(label32)
    label32 = label32.squeeze()
    pred32 = module(input32.unsqueeze(0)).squeeze().detach()
    
    print(input32.size(), label32.size())
    print(pred32.size())
    
    label = torch.cat((input32, label32), dim=0)
    pred = torch.cat((input32, pred32), dim=0)
    
    print(label.size())
    print(pred.size())
    
    vmin = label.min()
    vmax = label.max()
    
    def update(frame, axarr, pred, label):
        vorticity_cmap = seaborn.color_palette('icefire', as_cmap=True)
        error_cmap = seaborn.color_palette('mako', as_cmap=True)
        axarr[0].imshow(pred[frame], interpolation='spline16', cmap=vorticity_cmap)
        axarr[1].imshow(label[frame], interpolation='spline16', cmap=vorticity_cmap)
        abs_err = abs(pred[frame] - label[frame])
        axarr[2].imshow(abs_err, interpolation='spline16', cmap=error_cmap, vmin=0, vmax=abs_err.max())
        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

    fig, axarr = plt.subplots(1, 3)
    update_func = functools.partial(update, axarr=axarr, pred=pred, label=label)
    anim = ani.FuncAnimation(fig, update_func, frames=pred.shape[0], interval=500)
    anim.save('ns.gif', writer='imagemagick')
    plt.close()
    
    
    fig, axarr = plt.subplots(2, label.size(0), layout='constrained', figsize=(7, 4))
    vorticity_cmap = seaborn.color_palette('icefire', as_cmap=True)
    for i in range(label.size(0)):
        axarr[0, i].imshow(label[i], cmap=vorticity_cmap)
    for i in range(pred32.size(0)):
        im = axarr[1, input32.size(0) + i].imshow(pred32[i], cmap=vorticity_cmap)
    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.colorbar(im, ax=axarr[:, 3], ticks=[-3, 0, 3], fraction=0.05) #format="%4.0e")
    
    axarr[0, 0].title.set_text('Input 1')
    axarr[0, 1].title.set_text('Input 2')
    axarr[0, 2].title.set_text('Expected 1')
    axarr[0, 3].title.set_text('Expected 2')
    
    axarr[1, 2].title.set_text('Predicted 1')
    axarr[1, 3].title.set_text('Predicted 2')
    
    fig.delaxes(axarr[1, 0])
    fig.delaxes(axarr[1, 1])
    
    print(module.model)
        
    plt.savefig('ns_figs/ns_decaying.pdf')
    plt.close()
    
    fig, axarr = plt.subplots(1, 3, layout='constrained')
    
    def fft(data):
        return torch.log(abs(torch.fft.fftshift(torch.fft.fft2(data))))
        
    pred_fft = fft(pred32[-1])
    label_fft = fft(label32[-1])
    
    axarr[0].imshow(pred_fft, vmin=label_fft.min(), vmax=label_fft.max())
    axarr[1].imshow(label_fft)
    axarr[2].imshow(abs(pred_fft - label_fft) / label_fft)
    
    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.savefig('ns_figs/fft.png')
    plt.close()
    
    fig, axarr = plt.subplots(1, 2)
    axarr[0].plot(abs(torch.fft.fft2(pred32[-1])).mean(dim=0)[1:63], label='Predicted')
    axarr[0].plot(abs(torch.fft.fft2(label32[-1])).mean(dim=0)[1:63], label='Expected')
    axarr[0].set_title('Mean x-frequencies')
    axarr[0].set_ylabel('Amplitude')
    axarr[0].set_xlabel('Frequency')
    axarr[0].legend()
    
    axarr[1].plot(abs(torch.fft.fft2(pred32[-1])).mean(dim=1)[1:63], label='Predicted')
    axarr[1].plot(abs(torch.fft.fft2(label32[-1])).mean(dim=1)[1:63], label='Expected')
    axarr[1].set_title('Mean y-frequencies')
    axarr[1].set_xlabel('Frequency')
    axarr[1].legend()
    
    plt.savefig('ns_figs/mean_fft.png')
    plt.close()
    
    
if __name__ == "__main__":
    main()
