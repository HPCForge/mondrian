import torch
import torch.nn as nn
import torchdeq
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import ctypes

from torch import autograd
from torchdeq import reset_norm, apply_norm

debug = False

from mondrian_lib.fdm.deq_dd_fno import DDFNO
from neuralop.layers.mlp import MLP
from torchdeq.core import get_deq, register_deq, get_solver

from mondrian_lib.fdm.commons import SpectralConv2d, MLP2d


# This is using code from the FNO-DEQ Paper
class BasicBlock(nn.Module):
    # Note: parametrizing with a single layer-- might have to use more layers if this doesn't work well
    def __init__(self, modes1, modes2, width, add_mlp=False, normalize=False, activation=F.gelu):
        super(BasicBlock, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.add_mlp = add_mlp

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.normalize = normalize
        self.act = activation

        if add_mlp:
            self.mlp0 = MLP2d(self.width, self.width, self.width)
            self.mlp1 = MLP2d(self.width, self.width, self.width)
            self.mlp2 = MLP2d(self.width, self.width, self.width)

        if normalize:
            self.norm = nn.InstanceNorm2d(self.width)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

    def forward(self, x, injection=None):
        x = self.norm(x) if self.normalize else x

        x1 = self.conv0(x)
        x1 = self.norm(x1) if self.normalize else x1

        if self.add_mlp:
            x1 = self.mlp0(x1)

        x1 = self.norm(x1) if self.normalize else x1

        x2 = self.w0(x)
        x2 = self.norm(x2) if self.normalize else x2
        x = x1 + x2 + injection
        x = self.act(x)

        x1 = self.conv1(x)
        x1 = self.norm(x1) if self.normalize else x1

        if self.add_mlp:
            x1 = self.mlp1(x1)

        x1 = self.norm(x1) if self.normalize else x1

        x2 = self.w1(x)
        x2 = self.norm(x2) if self.normalize else x2
        x = x1 + x2 + injection
        x = self.act(x)

        x1 = self.conv2(x)
        x1 = self.norm(x1) if self.normalize else x1
        if self.add_mlp:
            x1 = self.mlp2(x1)
        x1 = self.norm(x1) if self.normalize else x1

        x2 = self.w2(x)
        x2 = self.norm(x2) if self.normalize else x2
        x = x1 + x2 + injection
        x = self.act(x)
        return x


class StackedBasicBlock(nn.Module):
    def __init__(self, modes1, modes2, width, depth=1, add_mlp=False, normalize=False):
        super(StackedBasicBlock, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.add_mlp = add_mlp
        self.depth = depth

        blocks = []
        for _ in range(depth):
            blocks.append(BasicBlock(self.modes1, self.modes2, self.width, add_mlp=add_mlp, normalize=normalize))

        self.deq_block = nn.ModuleList(blocks)

    def forward(self, x, injection=None):
        for idx in range(self.depth):
            x = self.deq_block[idx](x, injection)
        return x


# This is using code from the FNO-DEQ Paper
class DEQ_DDFNO(nn.Module):
    """
    CTOR:
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 pretraining_steps,
                 pretrain_iter_steps,
                 projected_gradient,
                 pg_steps,
                 f_max_iter,
                 f_thresh,
                 b_max_iter,
                 b_thresh,
                 device,
                 n_modes):

        super().__init__()
        self.n_dim = len(n_modes)
        self.tau = 0.5
        self.solver = get_solver("anderson")
        self.pretraining_steps = pretraining_steps
        self.pretrain_iter_steps = pretrain_iter_steps
        self.f_max_iter = f_max_iter
        self.b_max_iter = b_max_iter
        self.device = device
        self.projected_gradient = projected_gradient
        self.pg_steps = pg_steps

        self.train_steps = 0

        self.f_solver = self.solver
        self.f_thresh = f_thresh
        self.b_solver = self.solver
        self.b_thresh = b_thresh

        self.hidden_channels = 64

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = SpectralConv2d(self.hidden_channels, self.hidden_channels, n_modes[0], n_modes[1])
        self.w0 = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1)

        self.fnodeq = StackedBasicBlock(modes1=n_modes[0],
                                        modes2=n_modes[1],
                                        width=self.hidden_channels,
                                        depth=3,
                                        add_mlp=False,
                                        normalize=False)

        # self.ddfno = DDFNO(self.hidden_channels, self.hidden_channels, n_modes).float().to(self.device)

        self.deq_mode = False

        self.m1 = MLP(in_channels=self.in_channels,
                      out_channels=self.hidden_channels,
                      hidden_channels=self.hidden_channels,
                      n_layers=2,
                      n_dim=self.n_dim)

        self.m2 = MLP(in_channels=self.hidden_channels,
                      out_channels=self.out_channels,
                      hidden_channels=self.hidden_channels,
                      n_layers=2,
                      n_dim=self.n_dim)

    def forward(self, x):

        self.deq_mode = False
        if self.train_steps < 0 or self.train_steps >= self.pretraining_steps:
            self.deq_mode = True

        x = self.m1(x)

        # def f(_x): return self.ddfno(_x, xlim, ylim, injection=x)

        def f(z):
            return self.fnodeq(z, x)

        z1 = torch.zeros_like(x)

        if self.deq_mode:
            with torch.no_grad():
                re = self.f_solver(f,
                                   z1,
                                   name="forward",
                                   stop_mode='abs',
                                   eps=1e-3,
                                   f_thresh=self.f_thresh)
                z1 = re[0]
                next_z1 = z1
        else:
            for _ in range(self.pretrain_iter_steps):
                next_z1 = f(z1)
                abs_diff = (next_z1 - z1).norm().item()
                rel_diff = abs_diff / (1e-5 + z1.norm().item())
                z1 = next_z1
            next_z1 = z1

            if self.training:
                if self.projected_gradient:
                    def bhook(grad):
                        if self.hook is not None:
                            self.hook.remove()
                            torch.cuda.synchronize()
                        result = self.b_solver(lambda y: autograd.grad(next_z1, z1, y, retain_graph=True)[0] + grad,
                                               torch.zeros_like(grad),
                                               name="backward",
                                               eps=1e-3,
                                               threshold=self.b_thresh)
                        return result[0]

                    self.hook = next_z1.register_hook(bhook)
                    z1.requires_grad_()
                    for _ in range(self.pg_steps):
                        z1 = (1 - self.tau) * z1 + self.tau * f(z1)
                    next_z1 = z1
                else:
                    next_z1 = f(z1.requires_grad_())

        x1 = self.conv(next_z1)
        x2 = self.w0(next_z1)
        x = x1 + x2
        x = self.m2(x)

        if debug: print("DEQ_DDFNO: Forward() Return")
        return x
