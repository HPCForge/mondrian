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
        op_xlim: float,
        op_ylim: float,
        overlap: float,
        domain_padding:float ,
        use_coarse_op: bool
    ):
        super().__init__()
        self.layer = layer

        self.op_xlim = op_xlim
        self.op_ylim = op_ylim
        self.overlap = overlap

        CoarseOp = CoarseOpCNN
        self.c = CoarseOp(1, hc, hc, hc, op_xlim, op_ylim) 

    def _apply_op(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, t, global_xlim, global_ylim):
        # t should be laid out [batch, channel, y, x]
        assert t.dim() == 4

        if self.op_xlim == global_xlim and self.op_ylim == global_ylim:
            return self.layer(t)

        x_idx, y_idx, res_per_x, res_per_y = get_subdomain_indices(
                self.overlap,
                t.size(3),
                t.size(2),
                self.op_xlim,
                self.op_ylim,
                global_xlim,
                global_ylim)

        return self._apply_op(t, x_idx, y_idx, res_per_x, res_per_y, global_xlim, global_ylim)
