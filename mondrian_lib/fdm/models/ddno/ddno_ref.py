import torch
import torch.nn as nn
import torch.nn.functional as F

from mondrian_lib.fdm.subdomain_project import ToReference, FromReference

from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.mlp import MLP
from neuralop.layers.skip_connections import skip_connection
from mondrian_lib.fdm.models.ddno.subdomain_padding import SubDomainPadding
from mondrian_lib.fdm.models.ddno.self_attention import SelfAttention
from mondrian_lib.fdm.models.ddno.encoder import Encoder


class DDNORef(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        layers: int,
        n_subdomain_x: float,
        n_subdomain_y: float,
        lifting_channels=256,
        projection_channels=256,
    ):
        super().__init__()
        self.n_dim = 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.layers = layers
        self.padding = SubDomainPadding((0.2, 0.2))

        self.n_subdomain_x = n_subdomain_x
        self.n_subdomain_y = n_subdomain_y
        self.n_subdomains = n_subdomain_x * n_subdomain_y

        self.lifting = MLP(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
        )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
        )

        self.to_reference = nn.ModuleList([
                ToReference(self.hidden_channels,
                            self.hidden_channels,
                            self.hidden_channels)
                for _ in range(self.layers)
            ])
        self.from_reference = nn.ModuleList([
                FromReference(self.hidden_channels,
                              self.hidden_channels,
                              self.hidden_channels)
                for _ in range(self.layers)
            ])

        self.encoder = nn.ModuleList([
                Encoder(self.n_subdomains,
                        self.hidden_channels,
                        heads=1)
                for _ in range(self.layers)
            ])

    def _build_subdomain_lookup(self, t, coords):
        r""" This constructs a simple lookup table for subdomains.
        If lookup[:, i, j] == k, then index i,j is in subdomain k.
        Note, not all subdomains are the same size, as the subdomains
        along the outer edge may be slightly larger
        """
        x_res = t.size(3)
        y_res = t.size(2)
        lookup = torch.empty((y_res, x_res), device=t.device).int()

        subdomain_size_x = 2 / self.n_subdomain_x
        subdomain_size_y = 2 / self.n_subdomain_y

        start_y = -1
        for row in range(self.n_subdomain_y):
            start_x = -1
            for col in range(self.n_subdomain_x):
                end_y = start_y + subdomain_size_y
                end_x = start_x + subdomain_size_x

                if row == self.n_subdomain_y - 1:
                    end_y = float('inf')
                if col == self.n_subdomain_x - 1:
                    end_x = float('inf')

                x_coords = coords[0]
                y_coords = coords[1]
                x_mask = (x_coords >= start_x) & (x_coords < end_x)
                y_mask = (y_coords >= start_y) & (y_coords < end_y)

                subdomain_id = row * self.n_subdomain_x + col
                lookup[x_mask & y_mask] = subdomain_id 
                start_x += subdomain_size_x
            start_y += subdomain_size_y

        return lookup

    def forward(self, t):
        assert t.dim() == 4
        x_res = t.size(3)
        y_res = t.size(2)

        x_coords = torch.linspace(-1, 1, x_res + 2, device=t.device)[1:-1]
        y_coords = torch.linspace(-1, 1, y_res + 2, device=t.device)[1:-1]
        coords = torch.stack(torch.meshgrid(x_coords, y_coords, indexing='xy'), dim=0)

        subdomain_lookup = self._build_subdomain_lookup(t, coords)

        t = self.lifting(t)
        r = F.gelu(self.to_reference[0](t, coords, subdomain_lookup))

        self.padding.pad(r)

        for idx in range(self.layers):
            r = F.gelu(self.encoder[idx](r))

        t = t = F.gelu(self.from_reference[0](r, coords, subdomain_lookup))
        t = self.projection(t)
        return t
