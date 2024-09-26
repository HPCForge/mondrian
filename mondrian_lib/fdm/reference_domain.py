import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLP
from neuralop.layers.attention_kernel_integral import AttentionKernelIntegral
import einops

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
        projection_channels = 256
        self.n_dim = 2

        self.kernel = MLP(
            in_channels=1 + self.n_dim + self.in_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=self.n_dim)

        self.aki = AttentionKernelIntegral(
                self.in_channels,
                self.out_channels,
                n_heads=2,
                head_n_channels=2,
                pos_dim=2)

    def forward(self,
                v,
                physical_coords,
                subdomain_lookup):
        r"""
        project non-overlapping subfunctions of the input v to reference
        domains on [-1, 1]^2
        """
        assert v.dim() == 4
        assert physical_coords.dim() == 3
        assert subdomain_lookup.dim() == 2
        batch_size = v.size(0)
        x_res = v.size(3)
        y_res = v.size(2)

        reference_res = x_res * y_res
        reference_coords = torch.linspace(-1, 1, reference_res, device=v.device)

        n_subdomains = subdomain_lookup.max() + 1
        reference = torch.empty((batch_size, n_subdomains, self.out_channels, reference_res),
                                device=v.device)

        for idx in range(n_subdomains):
            mask = subdomain_lookup == idx
            # [batch, c, J_p]
            v_sub = v[:, :, mask]
            # [2, J_p]
            p_sub = physical_coords[:, mask]
            
            # sample random points in subdomain for each point in reference
            sample_size = 12
            size=(reference_coords.size(0), sample_size)
            sample_indices = torch.randint(low=0,
                                           high=v_sub.size(2) - 1,
                                           size=size,
                                           device=v.device)
            v_sub_sample = v_sub.\
                    unsqueeze(2).\
                    expand(-1, -1, reference_coords.size(0), -1).\
                    gather(dim=3, index=sample_indices.expand(batch_size, self.in_channels, -1, -1))
            p_sub_sample = p_sub.\
                    unsqueeze(1).\
                    expand(-1, reference_coords.size(0), -1).\
                    gather(dim=2, index=sample_indices.expand(2, -1, -1))

            # [1, reference_res, sample_size]
            rr = reference_coords.unsqueeze(1).repeat(1, sample_size).unsqueeze(0)
            # [3, reference_res, sample_size]
            coords = torch.cat((p_sub_sample, rr), dim=0)

            input = torch.cat((coords.expand(batch_size, -1, -1, -1),
                              v_sub_sample), dim=1)

            before_int = self.kernel(input)
            subdomain_transform = before_int.sum(3) / sample_size

            reference[:, idx] = subdomain_transform
        return reference
 
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
        # takes coordinates as input, output is a matrix
        self.kernel = MLP(
            in_channels=1 + self.n_dim + self.in_channels,
            out_channels=out_channels,
            hidden_channels=self.hidden_channels,
            n_layers=3,
            n_dim=self.n_dim)

    def forward(self,
                reference,
                physical_coords,
                subdomain_lookup):
        r"""
        project non-overlapping subfunctions of the input v to reference
        domains on [-1, 1]
        """
        assert reference.dim() == 4
        assert physical_coords.dim() == 3
        assert subdomain_lookup.dim() == 2
        batch_size = reference.size(0)
        x_res = physical_coords.size(2)
        y_res = physical_coords.size(1)
        reference_res = reference.size(3) 
        reference_coords = torch.linspace(-1, 1, reference_res, device=reference.device)

        s = subdomain_lookup.max()
        u = torch.empty(batch_size, self.out_channels, y_res, x_res, device=reference.device)
        for idx in range(s):
            mask = subdomain_lookup == idx
            # [batch, in_channels, reference_res]
            reference_domain = reference[:, idx]
            # [2, J_p]
            p_sub = physical_coords[:, mask]

            # sample random points in reference for each point in subdomain
            sample_size = 12
            size=(p_sub.size(1), sample_size)
            sample_indices = torch.randint(low=0,
                                           high=reference_coords.size(0),
                                           size=size,
                                           device=reference.device)
            # [batch, c, J_p, sample_size]
            reference_domain_sample = reference_domain.\
                    unsqueeze(2).\
                    expand(-1, -1, p_sub.size(1), -1).\
                    gather(dim=3, index=sample_indices.expand(batch_size, self.in_channels, -1, -1))
            # [J_p, sample_size]
            reference_coords_sample = reference_coords.\
                    unsqueeze(0).\
                    expand(p_sub.size(1), -1).\
                    gather(dim=1, index=sample_indices)

            # [3, J_p, sample_size]
            coords = torch.cat((reference_coords_sample.unsqueeze(0),
                                p_sub.unsqueeze(2).expand(-1, -1, sample_size)), dim=0)
            # [batch, out_channels, J_p, sample_size]
            input = torch.cat((coords.unsqueeze(0).expand(batch_size, -1, -1, -1),
                               reference_domain_sample), dim=1)
            u_sub = self.kernel(input).sum(3) / sample_size

            # write into expected subdomain
            u[:, :, mask] = u_sub 

        return u
