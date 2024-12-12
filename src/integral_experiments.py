from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt


def left_reimann_2d(f, dx):
    r"""
    Technically, a left Reimann sum should exclude the right-most part...
    """
    return (dx**2 * f[..., 0:-1, 0:-1]).sum(dim=(-2, -1))


def trapezoid_2d(f, dx):
    r"""
    weights  [1/4 1/2 1/2 ... 1/2 1/4]
             [1/2 1   1   ... 1   1/2]
             [            ...        ]
             [1/4 1/2 1/2 ... 1/2 1/4]
    """
    dxdx = dx**2
    inner_sum = (dxdx * f[..., 1:-1, 1:-1]).sum(dim=(-2, -1))

    y_weights = torch.full(f.size(-2), 1 / 2, device=f.device, dtype=f.dtype)
    y_weights[[0, -1]] = 1 / 4
    left_sum = (dxdx * f[..., 1:-1, 0] * y_weights).sum(dim=-1)
    right_sum = (dxdx * f[..., 1:-1, -1] * y_weights).sum(dim=-1)

    x_weights = torch.full(f.size(-1), 1 / 2, device=f.device, dtype=f.dtype)
    x_weights[[0, -1]] = 1 / 4
    top_sum = (dxdx * f[..., 0, 1:-1] * x_weights).sum(dim=-1)
    bottom_sum = (dxdx * f[..., -1, 1:-1] * x_weights).sum(dim=-1)

    return inner_sum + left_sum + right_sum + top_sum + bottom_sum


def simpsons_2d(f, dx):
    r"""
    simpsons rule uses triples of grid points, but we assume
    the grid input is a nice power of two. So, this uses a simpsons
    rule on the top-left block and a trapezoid rule on the bottom and
    right edges.
    weights []
    """
    pass


def reimann_left_1d(f, dx):
    return (f[:, :-1] * dx).sum(1)


def reimann_right_1d(f, dx):
    return (f[:, 1:] * dx).sum(1)


def reimann_1d(f, dx):
    return (f[:] * dx).sum(1)


def trapezoid_1d(f, dx):
    fd = f * dx
    return fd[:, 1:-1].sum(1) + (fd[:, 0] + fd[:, -1]) / 2


def simps_third_1d(f, dx):
    r"""
    Simpsons 1/3 rule can only be applied to an odd number of sub-intervals.
    """
    assert f.size(1) % 2 == 1
    fd = f * (dx / 3)
    weights = torch.zeros(fd.size(), dtype=f.dtype, device=f.device)
    weights[:, 1:-1:2] = 4
    weights[:, 2:-1:2] = 2
    weights[:, 0] = 1
    weights[:, -1] = 1
    fd = weights * fd
    return (fd).sum(1)


def simps_three_eighth_1d(f, dx):
    # assert f.size(1) % == 0
    fd = f * (3 / 8 * dx)
    weights = torch.zeros(fd.size(), dtype=f.dtype, device=f.device)
    weights[:, 1:-1:3] = 3
    weights[:, 2:-1:3] = 3
    weights[:, 3:-1:3] = 2
    weights[:, 0] = 1
    weights[:, -1] = 1
    fd = weights * fd
    return (fd).sum(1)


@dataclass
class IntegralData:
    grid_size: int
    rl_out: float
    rl_err: float

    r_out: float
    r_err: float

    t_out: float
    t_err: float

    s_out: float
    s_err: float


def test1(grid_size: int, dtype, device):
    # example taken from Kat Kinson, Chapter 5
    x = torch.linspace(0, torch.pi, d + 1, dtype=dtype, device=device).unsqueeze(0)
    x2 = torch.exp(x) * torch.cos(x)
    true = -12.0703463164

    dx = (x[:, 1] - x[:, 0])[0].item()
    reimann_left_x2 = reimann_left_1d(x2, dx)[0].item()
    reimann_x2 = reimann_1d(x2, dx)[0].item()
    trapezoid_x2 = trapezoid_1d(x2, dx=dx)[0].item()
    simps_x2 = simps_third_1d(x2, dx)[0].item()

    return IntegralData(
        grid_size=grid_size,
        rl_out=reimann_left_x2,
        rl_err=abs(reimann_left_x2 - true),
        r_out=reimann_x2,
        r_err=abs(reimann_x2 - true),
        t_out=trapezoid_x2,
        t_err=abs(trapezoid_x2 - true),
        s_out=simps_x2,
        s_err=abs(simps_x2 - true),
    )


device = "cuda"
dtype = torch.float16
print(device, dtype)

rl = []
r = []
t = []
s = []

for d in [2, 4, 6, 8, 16, 32, 64, 128, 256, 512]:
    integral = test1(d, dtype, device)
    rl.append((d, dtype, integral.rl_out, integral.rl_err, "reimmain_left"))
    r.append((d, dtype, integral.r_out, integral.r_err, "reimann"))
    t.append((d, dtype, integral.t_out, integral.t_err, "trapezoid"))
    s.append((d, dtype, integral.s_out, integral.s_err, "simpsons"))


def plot_integral(data):
    d, dtype, out, err, name = zip(*data)
    print(name)
    plt.plot(d, err, label=f"{name[0]}/{dtype[0]}")
    plt.yscale("log")


plot_integral(rl)
plot_integral(r)
plot_integral(t)
plot_integral(s)
plt.legend()
plt.savefig("err.png")
