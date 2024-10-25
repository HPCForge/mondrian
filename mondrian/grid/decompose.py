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