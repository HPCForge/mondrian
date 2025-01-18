import einops
import torch
from torch import nn

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from mondrian.grid.utility import cell_centered_unit_grid

class ProjectToNewGrid2d(nn.Module):
    r"""
    This uses cross-attention to project to a different discretization.
    """
    
    def __init__(self,
                 embed_dim,
                 num_heads,
                 scale=None):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = scale
        self.q_operator = nn.Linear(2, embed_dim, bias=False)
        self.kv_operator = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.out_operator = nn.Linear(embed_dim, embed_dim, bias=True)
    
    def _kv(self, f):
        f = einops.rearrange(f, 'b s c h w -> (b s) (h w) c')
        kv = self.kv_operator(f)
        k, v = einops.rearrange(kv, 
                                'batch_seq points (split num_heads head_dim) -> split batch_seq num_heads points head_dim',
                                split=2,
                                num_heads=self.num_heads,
                                head_dim=self.head_dim)
        return k, v
    
    def forward(self, f, scale=None, coords=None):
        assert f.dim() == 5
        assert f.size(-3) == self.embed_dim
        batch_size = f.size(0)
        seq_len = f.size(1)
        height = f.size(-2)
        width = f.size(-1)
        
        # check scale or coords specified, but not both.
        scale_input = scale is not None
        coords_input = coords is not None
        assert scale_input != coords_input
        
        # caller can specify a particular coordinate grid, or say to 
        if coords is None:
            target_height = self.scale * height
            target_width = self.scale * width
            coords = cell_centered_unit_grid((target_height, target_width), device=f.device)
        assert coords is not None
        assert coords.dim() == 3
        assert coords.size(-1) == 2

        
        query = self.q_operator(coords)
        key, value = self._kv(f)
        
        attn = scaled_dot_product_attention(query, key, value)
        attn = einops.rearrange(attn, 'bs heads points c -> bs points (heads c)')
        out = self.out_operator(attn)
        
        return einops.rearrange(out, '(b s) (h w) c -> b s c h w',
                                b=batch_size,
                                s=seq_len,
                                h=height,
                                w=width)