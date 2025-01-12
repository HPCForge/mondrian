import torch
import einops

from mondrian.attention.func_self_attention import FuncSelfAttention
from mondrian.attention.functional._naive import naive_func_attention
from mondrian.grid.quadrature import Quadrature2d


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


def test_func_sa_quadrature():
    v = torch.ones(8, 4, 32, 16, 16)

    r = naive_func_attention(v, v, v, Quadrature2d("reimann")(v))
    t = naive_func_attention(v, v, v, Quadrature2d("trapezoid")(v))
    s = naive_func_attention(v, v, v, Quadrature2d("simpson_13")(v))

    assert torch.allclose(s, t, atol=1e-4)

    x = torch.linspace(0, 2 * torch.pi, 16)
    x, y = torch.meshgrid(x, x, indexing="xy")

    data = torch.sin(x + y)
    data = torch.tile(data, (8, 4, 32, 1, 1))
    r = naive_func_attention(data, data, data, Quadrature2d("reimann")(data))
    t = naive_func_attention(data, data, data, Quadrature2d("trapezoid")(data))
    s = naive_func_attention(data, data, data, Quadrature2d("simpson_13")(data))

    assert torch.allclose(s, t, atol=1e-4)

    data = torch.sin(x * y + 3 * x + y**2)
    data = torch.tile(data, (8, 4, 32, 1, 1))
    r = naive_func_attention(data, data, data, Quadrature2d("reimann")(data))
    t = naive_func_attention(data, data, data, Quadrature2d("trapezoid")(data))
    s = naive_func_attention(data, data, data, Quadrature2d("simpson_13")(data))

    assert torch.allclose(r, t)
    assert torch.allclose(s, t, atol=1e-4)
    assert False


def test_func_sa_quadrature():
    def get_data(res):
        x = torch.linspace(0, 2 * torch.pi, res)
        x, y = torch.meshgrid(x, x, indexing="xy")
        data = torch.sin(x + y)
        data = torch.tile(data, (8, 4, 32, 1, 1))
        return data

    data = get_data(16)
    r16 = naive_func_attention(data, data, data, Quadrature2d("reimann")(data))
    t16 = naive_func_attention(data, data, data, Quadrature2d("trapezoid")(data))
    s16 = naive_func_attention(data, data, data, Quadrature2d("simpson_13")(data))

    data = get_data(128)
    r64, rscore = naive_func_attention(
        data, data, data, Quadrature2d("reimann")(data), return_scores=True
    )
    t64, tscore = naive_func_attention(
        data, data, data, Quadrature2d("trapezoid")(data), return_scores=True
    )
    s64, sscore = naive_func_attention(
        data, data, data, Quadrature2d("simpson_13")(data), return_scores=True
    )

    # assert torch.allclose(rscore, tscore)
    # assert torch.allclose(tscore, sscore)

    r16 = einops.rearrange(r16, "b s ... -> (b s) ...")
    t16 = einops.rearrange(t16, "b s ... -> (b s) ...")
    s16 = einops.rearrange(s16, "b s ... -> (b s) ...")

    r64 = einops.rearrange(r64, "b s ... -> (b s) ...")
    t64 = einops.rearrange(t64, "b s ... -> (b s) ...")
    s64 = einops.rearrange(s64, "b s ... -> (b s) ...")

    r16 = torch.nn.functional.interpolate(
        r16, size=(128, 128), mode="bicubic", align_corners=True
    )
    t16 = torch.nn.functional.interpolate(
        t16, size=(128, 128), mode="bicubic", align_corners=True
    )
    s16 = torch.nn.functional.interpolate(
        s16, size=(128, 128), mode="bicubic", align_corners=True
    )

    # assert ((r16 - r64) ** 2).max() < 1e-3
    # assert ((t16 - t64) ** 2).max() < 1e-3
    # assert ((s16 - s64) ** 2).max() < 1e-3
