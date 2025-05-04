import torch
import einops

from mondrian.attention.func_self_attention import FuncSelfAttention
from mondrian.attention.functional._naive import naive_func_attention
from mondrian.attention.func_self_attention import neighborhood_mask

"""
def test_vit_sa_init():
    vit = FuncSelfAttention(
        embed_dim=32,
        num_heads=4,
        head_split="channel",
        use_bias=False,
    )
    assert vit.embed_dim == 32
    assert vit.num_heads == 4
    assert vit.use_bias == False

    vit = FuncSelfAttention(
        embed_dim=32,
        num_heads=4,
        use_bias=True,
        head_split="channel",
    )
    assert vit.embed_dim == 32
    assert vit.num_heads == 4
    assert vit.use_bias == True


def test_vit_sa_forward_no_bias():
    vit = FuncSelfAttention(
        32, 4, use_bias=False, head_split="channel"
    )
    v = torch.ones(8, 4, 32, 16, 16)
    u = vit(v, 2, 2)
    assert u.size(0) == 8
    assert u.size(1) == 4
    assert u.size(2) == 32
    assert u.size(3) == 16
    assert u.size(4) == 16
    assert not u.isnan().any()


def test_vit_sa_forward_with_bias():
    vit = FuncSelfAttention(
        32, 4, use_bias=True, head_split="channel"
    )
    v = torch.ones(8, 4, 32, 16, 16)
    u = vit(v, 2, 2)
    assert u.size(0) == 8
    assert u.size(1) == 4
    assert u.size(2) == 32
    assert u.size(3) == 16
    assert u.size(4) == 16
    assert not u.isnan().any()
"""
    
def check_diagonal(mask):
    assert torch.all(torch.diag(mask) == True)
    assert torch.all(torch.tril(mask, diagonal=-1) == False)
    assert torch.all(torch.triu(mask, diagonal=1) == False) 

def test_neighborhood_mask_0_radius():
    # radius of zero should be diagonal mask
    mask = neighborhood_mask(1, 3, 0, 'cpu')
    assert mask.dim() == 2
    mask = neighborhood_mask(3, 1, 0, 'cpu')
    assert mask.dim() == 2
    
def test_neighborhood_mask_1_radius():
    # radius of 1 is banded, only corner values are False
    mask = neighborhood_mask(1, 3, 1, 'cpu')
    assert mask[2, 0] == False
    assert mask[0, 2] == False
    assert mask[2, 1] == True

def test_neighborhood_mask():
    mask = neighborhood_mask(3, 3, 1, 'cpu')
    for row in range(3):
        for col in range(3):
            # self should be included in self attention
            assert mask[row * 3 + col, row * 3 + col] == True
            # right neighbor
            if col < 2:
                assert mask[row * 3 + col, row * 3 + col + 1] == True
            # bottom-right neighbor
            if row < 2 and col < 2:
                assert mask[row * 3 + col, (row + 1) * 3 + col + 1] == True
            # top-left neighbor    
            if row > 0 and col > 0:
                assert mask[row * 3 + col, (row - 1) * 3 + col - 1] == True
            # bottom-right neighbor of bottom right neighbot is outside radius of 1
            if row < 1 and col < 1:
                assert mask[row * 3 + col, (row + 2) * 3 + col + 2] == False