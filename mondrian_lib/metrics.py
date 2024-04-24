import torch
import torch.nn.functional as F

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

class Metrics:
    def __init__(self, log_func):
        self.log_func = log_func

    def log_max_error(self, input, target, stage):
        max_err = max_error(input, target)
        self.log_func(f'{stage}/MaxError', l2_err.detach(), prog_bar=True)

    def log(self, input, target, stage):
        l2_err = l2_error(input, target)
        self.log_func(f'{stage}/L2Error', l2_err.detach(), prog_bar=True)

        l1_err = l1_error(input, target)
        self.log_func(f'{stage}/L1Error', l1_err.detach())

        return l2_err
