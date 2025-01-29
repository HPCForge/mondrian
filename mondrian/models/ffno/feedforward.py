from functools import partial
import einops
import torch.nn as nn


Linear = nn.Linear

class FeedForward(nn.Module):
    def __init__(self, dim, factor, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(
                nn.Sequential(
                    Linear(in_dim, out_dim),
                    nn.Dropout(dropout),
                    nn.GELU() if i < n_layers - 1 else nn.Identity(),
                    (
                        nn.LayerNorm(out_dim)
                        if layer_norm and i == n_layers - 1
                        else nn.Identity()
                    ),
                )
            )

    def forward(self, x):
        # reshaping for layernorm
        x = einops.rearrange(x, '... c h w -> ... h w c')
        for layer in self.layers:
            x = layer(x)
        x = einops.rearrange(x, '... h w c-> ... c h w')
        return x
