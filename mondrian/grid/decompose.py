import torch

def decompose2d(v, n_sub_x, n_sub_y):
  r"""
  Decomposes the input function `v` on a 2d domain into non-overlapping subdomains.
  Args:
    v: [batch x channels x y-discretization x x-discretization]
    n_sub_x: number of subdomains in the x direction
    n_sub_y: number of subdomains in the y direction.
  Returns:
    d: [batch x seq x channels x ...]
  """
  assert v.dim() == 4
  assert v.size(2) % n_sub_y == 0
  assert v.size(3) % n_sub_x == 0
  assert v.size(2) % 2 == 0
  assert v.size(3) % 2 == 0
  
  kernel_y = v.size(2) // n_sub_y
  kernel_x = v.size(3) // n_sub_x

  batch_size = v.size(0)
  channels = v.size(1)
  
  d = v.unfold(2, kernel_y, kernel_y) \
      .unfold(3, kernel_x, kernel_x) \
      .permute(0, 2, 3, 1, 4, 5) \
      .reshape(batch_size, -1, channels, kernel_y, kernel_x)
    
  return d

def recompose2d(d, n_sub_x, n_sub_y):
  r"""
  Recompose a sequence of functions, defined on subdomains, back to
  the original global domain. This should be the inverse of
  decompose2d.
  Args:
    d: [batch_size x seq x channels x ...]
  Returns: 
    v: [batch_size x channels x ...]
  """
  assert d.dim() == 5
  
  batch_size = d.size(0)
  channels = d.size(2)
  kernel_y = d.size(3)
  kernel_x = d.size(4)

  d = torch.unflatten(d, 1, (n_sub_y, n_sub_x))
  d = d.permute(0, 3, 4, 5, 1, 2)
  d = d.view(batch_size, -1, n_sub_y, n_sub_x).contiguous()
  d = d.view(batch_size, channels * kernel_y * kernel_x, -1).contiguous()     
  d = torch.nn.functional.fold(d, 
                               output_size=(n_sub_y * kernel_y, n_sub_x * kernel_x), 
                               kernel_size=(kernel_y, kernel_x),
                               stride=(kernel_y, kernel_x))
  
  return d