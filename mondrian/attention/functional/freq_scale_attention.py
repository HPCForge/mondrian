import math
import torch
import torch.nn.functional as F
import einops

def freq_scale_attention(query, key, value, freq_scale, attn_mask):
    r"""
    frequency-scaled attention from
    ANTI-OVERSMOOTHING IN DEEP VISION TRANSFORMERS. ICLR 2022.
    Args:
        query, key, value: [b, h, s, d, ...]
        freq_scale: [h, 1, 1], so it broadcasts on each head.
        attn_mask: [b, h, 1 ,1]
    """
    
    # assume inputs are sequence of 2d subdomains
    seq_len = query.size(2)
    dim = query.size(3)
    height = query.size(-2)
    width = query.size(-1)
    
    scores = einops.einsum(query, key, 'b h s1 ..., b h s2 ... -> b h s1 s2')
    probs = F.softmax(scores / (math.sqrt(dim) * height * width), dim=-1)
    
    # get portion of probs that acts like a high-pass filter and scale it.
    low_pass = (1 / seq_len)
    high_pass = probs - low_pass
    freq_scaled_scores = low_pass + (1 + freq_scale[:, None, None]) * high_pass
    freq_scaled_scores = freq_scaled_scores * attn_mask
    
    return einops.einsum(freq_scaled_scores, value, 'b h s1 s2, b h s2 ... -> b h s1 ...')