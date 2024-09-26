import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLP
from neuralop.layers.spectral_convolution import SpectralConv
import math
from mondrian_lib.fdm.kernel.full_rank_linear import FullRankLinearKernel
import einops

def _get_reference_coords(reference_res, batch_size, device):
    coords = torch.linspace(-1, 1, reference_res + 2, device=device)[1:-1]
    # [H, W, 2]
    coords = torch.stack(torch.meshgrid(coords, coords, indexing='xy'), dim=-1)
    return coords

def _setup_reference(reference_res,
                     batch_size,
                     n_subdomains,
                     out_channels,
                     device): 
    reference_coords = _get_reference_coords(reference_res, batch_size, device)
    size = (batch_size, n_subdomains, out_channels, reference_res, reference_res)
    reference = torch.empty(size, device=device)
    return reference, reference_coords

class ToReference(nn.Module):
    r"""
    Project subdomains of 2D grid to 1D reference domain
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_dim = 2

        #self.to_reference = FullRankLinearKernel(
        #        in_channels,
        #        out_channels,
        #        num_filters=8)

        self.to_reference = SpectralConv(
                in_channels,
                out_channels,
                n_modes=(16, 16))

    def forward(self,
                v,
                physical_coords,
                subdomain_lookup):
        r"""
        project non-overlapping subfunctions of the input v to reference
        domains onto a 2D domain [-1, 1]^2
        """
        assert v.dim() == 4
        assert physical_coords.dim() == 3
        assert subdomain_lookup.dim() == 2
        batch_size = v.size(0)
        x_res = v.size(3)
        y_res = v.size(2)
        n_subdomains = subdomain_lookup.max() + 1

        ref_res = x_res #// int(math.sqrt(n_subdomains))
        ref, ref_coords = _setup_reference(
                ref_res, batch_size, n_subdomains, self.out_channels, v.device)

        for idx in range(n_subdomains):
            mask = subdomain_lookup == idx
            v_sub = torch.where(mask, v, 0)
            u = self.to_reference(v_sub)
            ref[:, idx] = u
            #mask = torch.stack((mask, mask), dim=0)
            #p_sub = torch.where(mask, physical_coords, 0)
            #p_sub = einops.rearrange(p_sub, 'd h w -> h w d')
            #u = self.to_reference(v_sub, p_sub, ref_coords)
            #ref[:, idx] = u

        return ref

class FromReference(nn.Module):
    r"""
    Project 1D reference domains to subdomains of a 2D grid
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_dim = 2

        self.from_reference = SpectralConv(
                in_channels,
                out_channels,
                n_modes=(16, 16))

    def forward(self,
                reference,
                physical_coords,
                subdomain_lookup):
        r"""
        project non-overlapping subfunctions of the input v to reference
        domains on [-1, 1]^2
        """
        assert reference.dim() == 5
        assert physical_coords.dim() == 3
        assert subdomain_lookup.dim() == 2
        batch_size = reference.size(0)
        x_res = physical_coords.size(2)
        y_res = physical_coords.size(1)

        ref_res = reference.size(3) 
        ref_grid = _get_reference_coords(ref_res, batch_size, reference.device) 

        s = subdomain_lookup.max()
        u = torch.empty(batch_size, self.out_channels, y_res, x_res, device=reference.device)
        for idx in range(s):
            mask = subdomain_lookup == idx
            # [batch, in_channels, H, W]
            ref = reference[:, idx]
            # [batch, H, W, dim]
            u_sub = self.from_reference(ref) + ref
            u[:, :, subdomain_lookup == idx] = u_sub[:, :, subdomain_lookup == idx] 
            #mask = torch.stack((mask, mask), dim=0)
            #p_sub = torch.where(mask, physical_coords, 0)
            #p_sub = einops.rearrange(p_sub, 'd h w -> h w d')
            #u_sub = self.from_reference(ref, ref_grid, p_sub)
            # write into expected subdomain
            #u[:, :, subdomain_lookup == idx] = u_sub[:, :, subdomain_lookup == idx] 

        return u
