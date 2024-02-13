import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.mlp import MLP

class DDOp(nn.Module):
    """
    The Domain-Decomposed Operator
    """
    def __init__(
        self,
        layer,
        op_xlim,
        op_ylim,
        overlap):
        super().__init__()
        self.layer = layer

        # x, y sizes are all in spatial dimensions
        self.op_xlim = op_xlim
        self.op_ylim = op_ylim
        self.overlap = overlap

    def forward(self, t, global_xlim, global_ylim):
        # t should be laid out [batch, channel, y, x]
        assert t.dim() == 4
       
        # get the resolution per [1,1] box
        res_per_global_x = t.size(3) / global_xlim
        res_per_global_y = t.size(2) / global_ylim

        # get the resolution of each subdomain
        res_per_x = int(self.op_xlim * res_per_global_x)
        res_per_y = int(self.op_ylim * res_per_global_y)
        
        # get the stride of subdomains
        subdomain_step_x = int((1 - self.overlap) * res_per_x)
        subdomain_step_y = int((1 - self.overlap) * res_per_y)
        
        print(res_per_global_x, res_per_x, subdomain_step_x)

        x_idx = torch.arange(0, t.size(3), subdomain_step_x)
        y_idx = torch.arange(0, t.size(2), subdomain_step_y)

        # shift the subdomains on the edge into the interior
        x_idx[-1] = t.size(3) - res_per_x
        y_idx[-1] = t.size(2) - res_per_y
        x_idx[-2] -= res_per_x // 2
        y_idx[-2] -= res_per_x // 2
        print(res_per_x)
        print(x_idx)
        print(y_idx)

        # apply layer to each subdomain
        # TODO: remove the loops
        for x in x_idx:
            for y in y_idx:
                h = t[:,:,x:x+res_per_x,y:y+res_per_y]
                t[:,:,x:x+res_per_x,y:y+res_per_y] = self.layer(h)

        return t
