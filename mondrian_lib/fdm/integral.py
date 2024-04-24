import torch

def integral_2d(A, dx, dim1, dim2):
    r"""
    Compute a 2d Integratal of the tensor A along dim1 and dim2.
    This assumes a uniform grid-spacing in the x and y directions
    Args:
        A: torch.Tensor, [..., dim1, ..., dim2, ...]
        dx: grid spacing
        dim1: first dimension of A to integrate
        dim2: second dimension, greater than dim1
    """
    assert A.dim() >= 2
    assert dim1 < dim2
    int_dim1 = torch.trapezoid(A, dx=dx, dim=dim1)
    # subtract 1 from dim2, because dim1 was "removed" by the integral
    int_dim2 = torch.trapezoid(int_dim1, dx=dx, dim=dim2 - 1)
    return int_dim2
