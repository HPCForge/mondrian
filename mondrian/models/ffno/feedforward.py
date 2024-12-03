from functools import partial

import torch.nn as nn


Linear = partial(nn.Conv2d, kernel_size=(1, 1), stride=1, padding=0)

class FeedForward(nn.Module):
    def __init__(self, dim, factor, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                Linear(in_dim, out_dim),
                nn.Dropout(dropout),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers - 1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
