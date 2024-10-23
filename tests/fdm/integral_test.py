import torch

from mondrian_lib.fdm.integral import integral_2d

def test_integral_2d_1():
    A = torch.tensor([[0, 0], [0, 0]])
    assert integral_2d(A, dx=1, dim1=0, dim2=1) == 0
    A = A.unsqueeze(0) 
    assert integral_2d(A, dx=1, dim1=0, dim2=1)[0] == 0

def test_integral_2d_2():
    points = 100
    x = torch.linspace(0, 1, points)
    y = torch.linspace(0, 1, points)
    x, y = torch.meshgrid(x, y, indexing='xy')
    A = (2*x + 2*y)
    assert abs(integral_2d(A, dx=x[0, 1] - x[0, 0], dim1=0, dim2=1) - 2.0) <= 1e-16
    A = (1 - x**2 - y**2)
    assert abs(integral_2d(A, dx=x[0, 1] - x[0, 0], dim1=0, dim2=1) - 1/3) <= 1e-4
    A = torch.stack((2*x + 2*y, 1 - x**2 - y**2), dim=0)
    assert abs(integral_2d(A, dx=x[0, 1] - x[0, 0], dim1=1, dim2=2) - 2.0)[0] <= 1e-16
    assert abs(integral_2d(A, dx=x[0, 1] - x[0, 0], dim1=1, dim2=2) - 1/3)[1] <= 1e-4

def test_integral_2d_vector_value():
    points = 100
    x = torch.linspace(0, 1, points)
    y = torch.linspace(0, 1, points)
    x, y = torch.meshgrid(x, y, indexing='xy')
    A = torch.stack((2*x + 2*y, 1 - x**2 - y**2), dim=-1)
    # vector integral will just component-wise
    assert abs(integral_2d(A, dx=x[0, 1] - x[0, 0], dim1=0, dim2=1) - 2.0)[0] <= 1e-6
    assert abs(integral_2d(A, dx=x[0, 1] - x[0, 0], dim1=0, dim2=1) - 1/3)[1] <= 1e-4

def test_integral_affine_single_x():
    points = 100
    x = torch.linspace(0, 1, points)
    y = torch.linspace(0, 1, points)
    x, y = torch.meshgrid(x, y, indexing='xy')
    dx = x[0, 1] - x[0, 0]

    # [1, H, W, 2]
    A = torch.stack((2*x + 2*y, 1 - x**2 - y**2), dim=-1).unsqueeze(0)

    coords = torch.stack((x, y), dim=-1)

    # experiment with a single x value
    # 1. Standard version:
    B = torch.rand((2, 2))
    W = torch.rand((2, 2, 4))
    Wx = torch.einsum('mnd,hwd->mn', W[:, :, :2], torch.rand(2))
    Wy = torch.einsum('mnd,hwd->hwmn', W[:,:,2:], coords)
    WxA = torch.einsum('mn,bhwn->bhwm', B + Wx, A)
    WyA = torch.einsum('hwmn,bhwn->bhwm', Wy, A)

    WA = WxA + WyA
    standard = integral_2d(WA, dx=dx, dim1=1, dim2=2)

    # optimized version, pulling operation on x out
    b = integral_2d(A, dx=dx, dim1=1, dim2=2)
    ab = torch.einsum('mn,bn->bm', B + Wx, b)
    c = integral_2d(WyA, dx=dx, dim1=1, dim2=2)
    faster = ab + c

    assert torch.allclose(standard, faster)

def test_integral_affine_single_x():
    points = 100
    x = torch.linspace(0, 1, points)
    y = torch.linspace(0, 1, points)
    x, y = torch.meshgrid(x, y, indexing='xy')
    dx = x[0, 1] - x[0, 0]

    # [1, H, W, 2]
    A = torch.stack((2*x + 2*y, 1 - x**2 - y**2), dim=-1).unsqueeze(0)

    coords = torch.stack((x, y), dim=-1)

    # experiment with a batch of x values
    B = torch.rand((2, 2))
    W = torch.rand((2, 2, 4))
    Wx = torch.einsum('mnd,hwd->hwmn', W[:,:,:2], coords)
    Wy = torch.einsum('mnd,hwd->hwmn', W[:,:,2:], coords)
    BWx = B.unsqueeze(0).unsqueeze(0) + Wx
    WyA = torch.einsum('hwmn,bhwn->bhwm', Wy, A)

    # optimized version, pulling operation on x out
    b = integral_2d(A, dx=dx, dim1=1, dim2=2)
    ab = torch.einsum('hwmn,bn->bhwm', BWx, b)
    c = integral_2d(WyA, dx=dx, dim1=1, dim2=2)
    faster = ab + c

    # compare with result for particular x-value
    x = coords[0, 0]
    Wx = torch.einsum('mnd,d->mn', W[:, :, :2], x)
    Wy = torch.einsum('mnd,hwd->hwmn', W[:,:,2:], coords)
    WxA = torch.einsum('mn,bhwn->bhwm', B + Wx, A)
    WyA = torch.einsum('hwmn,bhwn->bhwm', Wy, A)

    WA = WxA + WyA
    standard00 = integral_2d(WA, dx=dx, dim1=1, dim2=2)

    # compare with result for particular x-value
    x = coords[10, 10]
    Wx = torch.einsum('mnd,d->mn', W[:, :, :2], x)
    Wy = torch.einsum('mnd,hwd->hwmn', W[:,:,2:], coords)
    WxA = torch.einsum('mn,bhwn->bhwm', B + Wx, A)
    WyA = torch.einsum('hwmn,bhwn->bhwm', Wy, A)

    WA = WxA + WyA
    standard1010 = integral_2d(WA, dx=dx, dim1=1, dim2=2)

    assert torch.allclose(faster[:, 0, 0], standard00)
    assert torch.allclose(faster[:, 10, 10], standard1010)
