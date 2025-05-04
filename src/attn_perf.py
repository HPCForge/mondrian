import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from mondrian.attention.functional.func_attention import func_attention
from mondrian.attention.functional.galerkin import galerkin_attention
from mondrian.grid.decompose import decompose2d

def flash_attention(query, key, value):
    r"""
    This can only be used when the inputs are half precision and head_dim <= 256
    """
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return scaled_dot_product_attention(
            query,
            key,
            value)
        
def math_attention(query, key, value):
    return scaled_dot_product_attention(
            query,
            key,
            value)
    
    
def time(query, key, value, method):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    z = method(query, key, value)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def time_method(q, k, v, method):
    for i in range(10):
        method(q, k, v)
    torch.cuda.synchronize()
    
    times = []
    for i in range(20):
        t = time(q, k, v, method)
        times.append(t)
    mean = sum(times) / len(times)
    var = torch.var(torch.tensor(times))
    return mean, var

def get_subdomain_data(
    batch_size,
    heads,
    seq_len,
    channels,
    sub_h,
    sub_w
):
    data = torch.randn(batch_size, heads, seq_len, channels, sub_h, sub_w)
    q = torch.randn_like(data)
    k = torch.randn_like(data)
    v = torch.randn_like(data)
    return q, k, v

def get_point_data(
    batch_size, 
    heads,
    seq_len,
    head_dim
):
    data = torch.randn(batch_size, heads, seq_len, head_dim)
    q = torch.randn_like(data)
    k = torch.randn_like(data)
    v = torch.randn_like(data)
    return q, k, v

point_config = {
    'batch_size': [8],
    'heads': [4, 8],
    # 32x32 through 128x128
    'seq_len': [256, 1024, 9216, 16384],
    'head_dim': [32, 64, 128, 256]
}

dd_config = {
    'batch_size': [8],
    'heads': [4],
    'seq_len': [
        # NxN grid broken into 8x8 subdomains
        (16, 8, 8),   # N=32
        (64, 8, 8),   # 64
        (256, 8, 8),  # 128
        (1024, 8, 8), # 256
        (4096, 8, 8), # 512
        # NxN grid broken into 16x16 subdomains
        (4, 16, 16),
        (16, 16, 16), 
        (64, 16, 16), 
        (256, 16, 16),
        (1024, 16, 16)
    ],
    # channels, since the head_dim is essentially channels * subdomain_pixels
    'channels': [4, 8, 16, 32, 64, 128, 256]
}

# get perf for point-wise attention
for batch_size in point_config['batch_size']:
    for heads in point_config['heads']:
        for seq_len in point_config['seq_len']:
            for head_dim in point_config['head_dim']:
                for method in [flash_attention, galerkin_attention]:
                    if method is flash_attention and seq_len > 4096:
                        continue
                    q, k, v = get_point_data(batch_size, heads, seq_len, head_dim)
                    q = q.bfloat16()
                    k = k.bfloat16()
                    v = v.bfloat16()
                    time_mean, time_var = time_method(q, k, v, flash_attention)
                    print(method, heads, seq_len, head_dim, time_mean, time_var)

"""
# get perf for subdomain attention
for batch_size in dd_config['batch_size']:
    for heads in dd_config['heads']:
        for (seq_len, sub_h, sub_w) in dd_config['seq_len']:
            for channels in dd_config['channels']:
                q, k, v = get_subdomain_data(batch_size, heads, seq_len, channels, sub_h, sub_w)
                time_mean, time_var = time_method(q, k, v, func_attention)
                print(heads, seq_len, channels, time_mean, time_var)
"""
