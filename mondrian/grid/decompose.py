import einops


def decompose2d(v, n_sub_x, n_sub_y):
    r"""
    Decomposes the input function `v` on a 2d domain into non-overlapping subdomains.
    Args:
      v: [... x channels x y-discretization x x-discretization]
      n_sub_x: number of subdomains in the x direction
      n_sub_y: number of subdomains in the y direction
    Returns:
      d: [... x seq x channels x y-sub x x-sub]
    """
    assert v.dim() >= 4
    assert v.size(2) % n_sub_y == 0
    assert v.size(3) % n_sub_x == 0
    assert v.size(2) % 2 == 0
    assert v.size(3) % 2 == 0
    d = einops.rearrange(
        v, "... c (h1 h) (w1 w) -> ... (h1 w1) c h w", h1=n_sub_y, w1=n_sub_x
    )
    return d


def recompose2d(d, n_sub_x, n_sub_y):
    r"""
    Recompose a sequence of functions, defined on subdomains, back to
    the original global domain. This should be the inverse of
    decompose2d.
    Args:
      d: [batch_size x seq x channels x ...]
      n_sub_x: number of subdomains in the x direction
      n_sub_y: number of subdomains in the y direction
    Returns:
      v: [batch_size x channels x ...]
    """
    assert d.dim() >= 5
    v = einops.rearrange(
        d, "... (h1 w1) c h w -> ... c (h1 h) (w1 w)", h1=n_sub_y, w1=n_sub_x
    )
    return v

def win_decompose2d(v, n_sub_x, n_sub_y, window_size):
  r"""
  Decompose the input fuction into windows and then into subdomains.
  Args:
    v: [batch x channels x y-discretization x x-discretization]
    n_sub_x: number of subdomains in the x direction
    n_sub_y: number of subdomains in the y direction
    window_size: size of each window in patches

  Returns:
  windows: [(n_windows * batch_size) x (seq_len // n_windows) x embed_dim x ...]
  """
  assert v.dim() == 4
  assert v.size(2) % n_sub_y == 0
  assert v.size(3) % n_sub_x == 0
  assert n_sub_x % window_size == 0
  assert n_sub_y % window_size == 0

  batch_size, embed_dim = v.shape[:2]
  n_windows = (n_sub_x * n_sub_y) // (window_size ** 2)
  patch_discret_x = v.size(-1) // n_sub_x
  patch_discret_y = v.size(-2) // n_sub_y
  win_discret_x = patch_discret_x * window_size
  win_discret_y = patch_discret_y * window_size
  
  sub_per_window_x = n_sub_x // window_size
  sub_per_window_y = n_sub_y // window_size
  
  # [batch, windows, embed_dim, height, width]
  win = einops.rearrange(v, 'b c (window1 h) (window2 w) -> b (window1 window2) c h w', window1=window_size, window2=window_size)
  win = einops.rearrange(win, '... c (h1 h) (w1 w) -> ... (h1 w1) c h w', h1=sub_per_window_y, w1=sub_per_window_x)
  
  # decompose the input function into windows
  win = decompose2d(v, n_sub_y // window_size, n_sub_x // window_size)
  win = win.view(batch_size * n_windows, -1, win_discret_y, win_discret_x)
  # decomose each window into subdomains
  win = decompose2d(win, window_size, window_size)

  return win

def win_recompose2d(win, n_sub_x, n_sub_y, window_size):
  r"""
  Recompose a sequence of functions, defined on subdomains, back to
  the original global domain. This should be the inverse of win_decompose2d.
  Args:
    win: [(n_windows * batch_size) x (seq_len // n_windows) x (heads * embed_dim) x ...]
    n_sub_x: number of subdomains in the x direction
    n_sub_y: number of subdomains in the y direction
    window_size: size of each window in patches

  Returns: 
    v: [batch_size x channels x ...]
  """
  batch_size, _ ,embed_dim = win.shape[:3]
  n_windows = (n_sub_x * n_sub_y) // (window_size ** 2)
  patch_discret_x = win.size(-1)
  patch_discret_y = win.size(-2)
  win_discret_x = patch_discret_x * window_size
  win_discret_y = patch_discret_y * window_size
  win_patch_num = n_sub_x // window_size
  
  # recompose to windows
  v = recompose2d(win, window_size, window_size)
  v = v.view(batch_size//n_windows, -1, embed_dim, win_discret_x, win_discret_y)
  # recompose to the original domain
  v = recompose2d(v, win_patch_num, win_patch_num)
  
  return v