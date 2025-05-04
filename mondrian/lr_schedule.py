import numpy as np
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR, 
    SequentialLR
)

def warmup(warmup_iters: int):
    return lambda current_iter: current_iter / warmup_iters


class LinearWarmup(LambdaLR):
    def __init__(self, optimizer, warmup_iters, last_epoch=-1):
        assert warmup_iters > 0
        super().__init__(optimizer, lr_lambda=warmup(warmup_iters), last_epoch=last_epoch)


class WarmupCosineAnnealingLR(SequentialLR):
    def __init__(self, optimizer, warmup_iters, total_iters, eta_min=0, last_epoch=-1):
        assert warmup_iters < total_iters
        warmup = LinearWarmup(optimizer, warmup_iters)
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_iters - warmup_iters, eta_min=eta_min
        )

        super().__init__(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_iters],
            last_epoch=last_epoch,
        )
        
        
class WarmupCosineAnnealingWarmRestartsLR(SequentialLR):
    def __init__(self, optimizer, warmup_iters, total_iters, eta_min=0, last_epoch=-1):
        assert warmup_iters < total_iters
        warmup = LinearWarmup(optimizer, warmup_iters)
        cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2, eta_min=eta_min
        )

        super().__init__(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_iters],
            last_epoch=last_epoch,
        )