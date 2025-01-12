
import torch

from mondrian.models.transformer.vit_operator_2d import ViTOperator2d
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity
)

def profile_vit_operator():
    model = ViTOperator2d(32, 32, 128, 64, 'channel', 'reimann', 4, 64, (1, 1)).cuda()
    
    data = torch.randn(8, 32, 128, 128, device='cuda')
    ds_x = 16
    ds_y = 16
    for _ in range(3):
       out = model(data, ds_x, ds_y)
       out.sum().backward()
        
    with profile(activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ]) as prof:
        for _ in range(10):
            with record_function("forward"):
                out = model(data, ds_x, ds_y)
            with record_function("backward"):
                out.sum().backward()

    prof.export_chrome_trace("trace.json")


def main():
    profile_vit_operator()

if __name__ == "__main__":
    main()