import glob

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .positional_encoding_module import GaussianFourierFeatureTransform
from .factorization_module import FABlock2D

from mondrian.grid.utility import cell_centered_unit_grid


class FactorizedTransformer(nn.Module):
    def __init__(self, dim, dim_head, heads, dim_out, depth, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(
                nn.Sequential(
                    GaussianFourierFeatureTransform(2, dim // 2, 1), nn.Linear(dim, dim)
                )
            )
            layer.append(
                FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True, **kwargs)
            )
            self.layers.append(layer)

    def forward(self, u, pos_lst):
        b, nx, ny, c = u.shape
        nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        pos = torch.stack(
            torch.meshgrid([pos_lst[0].squeeze(-1), pos_lst[1].squeeze(-1)]), dim=-1
        )
        for pos_enc, attn_layer in self.layers:
            u += pos_enc(pos).view(1, nx, ny, -1)
            a = attn_layer(u, pos_lst).reshape(b, nx, ny, c)
            u = u + a
        return u


class FactFormer2D(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 dim,
                 depth,
                 dim_head,
                 heads,
                 pos_in_dim,
                 pos_out_dim,
                 positional_embedding,
                 kernel_multiplier,
                 resolution=None,
):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dim = dim  # dimension of the transformer
        self.depth = depth  # depth of the encoder transformer
        self.dim_head = dim_head
        
        # NOTE: This is used in the upsample block...
        self.resolution = resolution

        self.heads = heads

        self.pos_in_dim = pos_in_dim
        self.pos_out_dim = pos_out_dim
        self.positional_embedding = positional_embedding
        self.kernel_multiplier = kernel_multiplier

        self.to_in = nn.Linear(self.in_dim, self.dim, bias=True)

        self.encoder = FactorizedTransformer(
            self.dim,
            self.dim_head,
            self.heads,
            self.dim,
            self.depth,
            kernel_multiplier=self.kernel_multiplier,
        )
        
        # only use up/down block when resolution is specified.
        if self.resolution is not None:
            self.down_block = nn.Sequential(
                nn.InstanceNorm2d(self.dim),
                nn.Conv2d(
                    self.dim, self.dim // 2, kernel_size=3, stride=2, padding=1, bias=True
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.dim // 2,
                    self.dim // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
            )

            self.up_block = nn.Sequential(
                nn.Upsample(size=(self.resolution, self.resolution), mode="nearest"),
                nn.Conv2d(
                    self.dim // 2,
                    self.dim // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.dim // 2, self.dim, kernel_size=3, stride=1, padding=1, bias=True
                ),
            )

        self.simple_to_out = nn.Sequential(
            Rearrange("b nx ny c -> b c (nx ny)"),
            nn.GroupNorm(num_groups=8, num_channels=self.dim * 2),
            nn.Conv1d(
                self.dim * 2, self.dim, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.GELU(),
            nn.Conv1d(
                self.dim, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(
        self,
        u,
        domain_size_x,
        domain_size_y,
    ):

        # In the original implementation, `pos_lst`` was passed in.
        # For consistency with the other model APIs, I am passing in the
        # domain sizes and making `pos_lst` here.
        height = u.size(-2)
        width = u.size(-1)
        x = 2 * cell_centered_unit_grid(
            (width,), device=u.device
        ).unsqueeze(-1) - 1
        y = 2 * cell_centered_unit_grid(
            (height,), device=u.device
        ).unsqueeze(-1) - 1
        pos_lst = [x, y]
        
        u = rearrange(u, "b c h w -> b h w c")
        b, nx, ny, c = u.shape
        u = self.to_in(u)
        u_last = self.encoder(u, pos_lst)
        
        # if an input resolution is specified, we can use
        # the up.down block.
        if self.resolution is not None:        
            u = rearrange(u_last, "b nx ny c -> b c nx ny")
            u = self.down_block(u)
            u = self.up_block(u)
            u = rearrange(u, "b c nx ny -> b nx ny c")
            
        u = torch.cat([u, u_last], dim=-1)
        u = self.simple_to_out(u)
        u = rearrange(u, "b c (nx ny) -> b c nx ny", nx=nx, ny=ny)
        return u
