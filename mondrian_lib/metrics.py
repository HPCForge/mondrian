import torch
import torch.nn.functional as F

def log_metrics(loss_func,
                input,
                target,
                logger,
                stage,
                on_step,
                prog_bar):
    loss = loss_func(input, target)
    self.logger(f'{stage}/Loss', loss, on_step=on_step, prog_bar=prog_bar)

    self.logger(f'{stage}/L1Loss', l1_loss)

    return loss

class Metrics:
    def __init__(self, log_func):
        self.log_func = log_func

    def l2_error(self, input, target):
        return F.mse_loss(input, target)

    def l1_error(self, input, target):
        return F.l1_loss(input, target)

    def log(self, input, target, stage):
        l2_error = self.l2_error(input, target)
        self.log_func(f'{stage}/L2Error', l2_error.detach(), prog_bar=True)

        l1_error = self.l1_error(input, target)
        self.log_func(f'{stage}/L1Error', l1_error.detach())

        return l2_error
