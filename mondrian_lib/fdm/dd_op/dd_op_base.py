import torch
import torch.nn as nn
from neuralop.layers.padding import DomainPadding
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.mlp import MLP
from mondrian_lib.fdm.get_subdomain_indices import get_subdomain_indices
from mondrian_lib.fdm.coarse_op.coarse_op_cnn import CoarseOpCNN

class DDOpBase(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        hc: int,
        domain_size_x: float,
        domain_size_y: float,
        subdomain_size_x: float,
        subdomain_size_y: float,
        overlap: float,
        domain_padding:float,
        use_coarse_op: bool
    ):
        super().__init__()
        self.layer = layer

        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        self.subdomain_size_x = subdomain_size_x
        self.subdomain_size_y = subdomain_size_y

        self.overlap = overlap

        self.use_coarse_op = use_coarse_op
        if self.use_coarse_op:
            CoarseOp = CoarseOpCNN
            self.c = CoarseOp(1, hc, hc, hc, subdomain_size_x, subdomain_size_y) 
            self.mlp_combine = MLP(2 * hc, hc, hc, n_layers=1, n_dim=2)

    def _apply_op(self, *args, **kwargs):
        raise NotImplementedError

    def _apply_coarse_op(self, t, h):
        c = self.c(t, self.domain_size_x, self.domain_size_y)
        h = torch.cat((h, c), dim=1)
        h = self.mlp_combine(h)
        return h

    def forward(self, t):
        # t should be laid out [batch, channel, y, x]
        assert t.dim() == 4

        if (self.subdomain_size_x == self.domain_size_x and
            self.subdomain_size_y == self.domain_size_y):
            return self.layer(t)

        x_idx, y_idx, res_per_x, res_per_y = get_subdomain_indices(
                self.overlap,
                t.size(3),
                t.size(2),
                self.domain_size_x,
                self.domain_size_y,
                self.subdomain_size_x,
                self.subdomain_size_y)

        h = self._apply_op(t, x_idx, y_idx, res_per_x, res_per_y)
        if self.use_coarse_op:
            h = self._apply_coarse_op(t, h)
        return h
