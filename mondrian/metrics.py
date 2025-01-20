import torch
import torch.nn.functional as F
from mondrian.grid.quadrature import simpsons_13_quadrature_weights


def l2_error(input, target):
    return F.mse_loss(input, target)


def l1_error(input, target):
    return F.l1_loss(input, target)


def max_error(input, target):
    r"""
    Point-wise max across channels and spatial dims.
    This doesn't make sense to use directly for bubbleml,
    since each field should probably be considered separately.
    """
    m = torch.max(abs(input - target), dim=[1, 2, 3])
    return torch.mean(m)

def func_l2_error(input, target):
    height = input.size(-2)
    width = input.size(-1)
    quadrature_weights = simpsons_13_quadrature_weights((1, 1), (height, width), device=input.device)
    
    # compute quadrature spatially and mean over channels for MSE
    # then reduce with overall mean.
    return (quadrature_weights * (input - target)**2).sum(dim=(-2, -1)).mean(-1).mean()


class Metrics:
    def __init__(self, log_func):
        self.log_func = log_func

    def log_max_error(self, input, target, stage):
        max_err = max_error(input, target)
        batch_size = input.size(0)
        self.log_func(
            f"{stage}/MaxError",
            max_err.detach(),
            prog_bar=True,
            batch_size=batch_size,
        )

    def log(self, input, target, stage):
        l2_err = l2_error(input, target)
        batch_size = input.size(0)
        self.log_func(
            f"{stage}/L2Error",
            l2_err.detach(),
            prog_bar=True,
            batch_size=batch_size,
        )

        l1_err = l1_error(input, target)
        self.log_func(f"{stage}/L1Error", l1_err.detach(), batch_size=batch_size)

        func_l2_err = func_l2_error(input, target)
        self.log_func(
            f"{stage}/FuncL2Error",
            func_l2_err.detach(),
            prog_bar=True,
            batch_size=batch_size
        )

        return func_l2_err
