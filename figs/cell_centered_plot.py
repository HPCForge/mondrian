r"""
This just makes a simple figure showing a decomposition of a cell-centered grid.
"""


import matplotlib.pyplot as plt
import numpy as np

def cell_centered(d):
    delta = 1 / d
    x = delta * (np.arange(0, d) + 0.5)
    y = x
    x, y = np.meshgrid(x, y)
    return x.flatten(), y.flatten()

def plot_cell_centered():
    fig, axarr = plt.subplots(1, 2, figsize=(8, 4))
    
    x, y = cell_centered(4)
    axarr[0].scatter(x, y, c='black')
    
    x, y = cell_centered(8)
    axarr[1].scatter(x, y, c='black')

    for i in range(len(axarr)):
        axarr[i].set_xlim([0, 1])
        axarr[i].set_ylim([0, 1])
        axarr[i].set_xticks([])
        axarr[i].set_yticks([])

    deltas = [1 / 4, 1 / 8]
    for i, delta in enumerate(deltas):
        for x in np.arange(delta, 1, delta):
            axarr[i].plot([x, x], [0, 1], linewidth=1, color='gray')
            cell_line, = axarr[i].plot([0, 1], [x, x], linewidth=2, color='gray')
    
    for i in range(len(axarr)):
        axarr[i].plot([0.5, 0.5], [0, 1], linewidth=2, color='purple')
        subdomain_line, = axarr[i].plot([0, 1], [0.5, 0.5], linewidth=2, color='purple')
                
    axarr[0].legend([cell_line, subdomain_line], ['Cell', 'Subdomain'], loc='upper left')
    
    plt.savefig("cell_centered.png", bbox_inches="tight")
    
plot_cell_centered()
