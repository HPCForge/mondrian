import torch

from mondrian.grid.seq_op import seq_op

def test_seq_op():
  r"""
  The flattening / unflattening should not copy data, so the
  input and output data pointers should be to the same location.
  """
  data = torch.randn(5, 5, 32, 32, 32)
  fn = lambda x: x
  out = seq_op(fn, data)
  assert data.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()