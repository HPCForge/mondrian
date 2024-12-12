import pytest
from util import available_devices

import torch

from mondrian.models.galerkin_transformer_2d import GalerkinTransformer2d


@pytest.mark.parametrize("device", available_devices())
def test_galerkin_transformer_forward(device):
    vo = GalerkinTransformer2d(
        in_channels=32, out_channels=32, embed_dim=16, num_heads=4, num_layers=4
    ).to(device)
    v = torch.ones(4, 32, 32, 32).to(device)
    u = vo(v, 1, 1)
    assert u.size() == v.size()
    assert torch.all(u.isfinite())
