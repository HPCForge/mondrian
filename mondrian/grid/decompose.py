import einops


def decompose2d(v, n_sub_x, n_sub_y):
    r"""
    Decomposes the input function `v` on a 2d domain into non-overlapping subdomains.
    Args:
      v: [batch x channels x y-discretization x x-discretization]
      n_sub_x: number of subdomains in the x direction
      n_sub_y: number of subdomains in the y direction
    Returns:
      d: [batch x seq x channels x ...]
    """
    assert v.dim() == 4
    assert v.size(2) % n_sub_y == 0
    assert v.size(3) % n_sub_x == 0
    assert v.size(2) % 2 == 0
    assert v.size(3) % 2 == 0
    d = einops.rearrange(
        v, "b c (h1 h) (w1 w) -> b (h1 w1) c h w", h1=n_sub_y, w1=n_sub_x
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
    assert d.dim() == 5
    v = einops.rearrange(
        d, "b (h1 w1) c h w -> b c (h1 h) (w1 w)", h1=n_sub_y, w1=n_sub_x
    )
    return v
