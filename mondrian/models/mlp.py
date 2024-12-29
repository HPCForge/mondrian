import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.add_layer(in_dim, hidden_dim)

        for _ in range(num_layers - 1):
            self.add_layer(hidden_dim, hidden_dim)

        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def add_layer(self, in_dim, out_dim):
        self.layers.append(nn.Linear(in_dim, out_dim))
        self.layers.append(nn.GELU())

    @torch.compile
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
