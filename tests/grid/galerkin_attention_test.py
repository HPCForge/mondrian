import torch
import pytest

from mondrian.grid.attention.triton.galerkin import (
    galerkin_attention as galerkin_attention_triton,
)
from mondrian.grid.attention.functional.galerkin import galerkin_attention


# helpful if a test only makes sense for GPU
def cuda_devices():
    if torch.cuda.is_available():
        return ["cuda"]
    return []


"""
@pytest.mark.parametrize("device", cuda_devices())
def test_galerkin_attention(device):
  query = torch.randn(1, 1, 16, 32, device=device, dtype=torch.float32)
  key = torch.ones(1, 1, 16, 32, device=device, dtype=torch.float32)
  value = torch.ones(1, 1, 16, 32, device=device, dtype=torch.float32)

  gat = galerkin_attention_triton(query, key, value)
  ga = galerkin_attention(query, key, value, None)
  
  print(gat)
  print(ga)
  #print(gat - ga)
  
  # what a reasonable precision for this?
  assert torch.allclose(gat, ga, atol=1)
"""
