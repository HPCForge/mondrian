import torch 

def seq_op(op, v):
  r"""
  Applies op to each batch / sequence item separately. This is
  needed because a lot of operations work on data shaped [B x C x H x W],
  so we need to flatten the batch and sequence dimensions to use these.
  Args:
    v: [batch x seq x ...]
  Returns:
    u: [batch x seq x ...]
  """
  batch_size = v.size(0)
  seq_len = v.size(1)
  v = torch.flatten(v, start_dim=0, end_dim=1)
  v = op(v)
  v = torch.unflatten(v, 0, (batch_size, seq_len))
  return v