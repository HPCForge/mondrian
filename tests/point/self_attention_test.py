import torch

from mondrian.point.self_attention.self_attention import SelfAttention


def test_init():
    sa = SelfAttention(in_channels=32, hidden_channels=32, heads=4)
    assert sa.in_channels == 32
    assert sa.heads == 4
    assert sa.Q.out_features == 4 * 32
    assert sa.Q.weight.size(0) == 4 * 32


def test_forward():
    sa = SelfAttention(in_channels=32, hidden_channels=32, heads=4)

    v = torch.randn(4, 16, 32)
    u = sa(v)
    assert v.size() == u.size()
