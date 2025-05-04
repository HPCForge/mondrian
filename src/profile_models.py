
import torch
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity
)

from mondrian.models.transformer.vit_operator_2d import ViTOperator2d, ViTOperatorCoarsen2d
from mondrian.grid.quadrature import set_default_quadrature_method
from mondrian.layers.qkv_operator import set_default_qkv_operator
from mondrian.layers.feed_forward_operator import set_default_feed_forward_operator
from mondrian.layers.spectral_conv import set_default_spectral_conv_modes


def profile_vit_operator():
    torch.set_float32_matmul_precision("high")
    
    # sets the default quadrature method for any integrals evaluated by the model
    set_default_quadrature_method('reimann')
    set_default_qkv_operator('separable_operator')
    set_default_feed_forward_operator('separable_neural_operator')
    
    model = ViTOperator2d(
        in_channels=1, 
        out_channels=1, 
        embed_dim=64, 
        channel_heads=4, 
        x_heads=1, 
        y_heads=1, 
        attn_neighborhood_radius=None,
        num_layers=4, 
        max_seq_len=64, 
        subdomain_size=(1, 1)).cuda()
    
    data = torch.randn(8, 1, 32, 32, device='cuda')
    ds_x = 8
    ds_y = 8
    for _ in range(3):
       out = model(data, ds_x, ds_y)
       out.sum().backward()
        
    with profile(activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ]) as prof:
        for _ in range(5):
            with record_function("forward"):
                out = model(data, ds_x, ds_y)
            with record_function("backward"):
                out.sum().backward()

    prof.export_chrome_trace("trace.json")


def main():
    profile_vit_operator()

if __name__ == "__main__":
    main()