r"""
Just double checking what different settings for interp do...
It is not very well documented...!
"""
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch_cubic_spline_grids import (
    CubicBSplineGrid2d,
    CubicBSplineGrid4d
)

import torch
from mondrian.grid.grid_interpolator import RegularGridInterpolator
import numpy as np

"""
points = [torch.arange(-.5, 2.5, .2) * 1., torch.arange(-.5, 2.5, .2) * 1.]
values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
gi = RegularGridInterpolator(points, values)

X, Y = np.meshgrid(np.arange(-.5, 2.5, .02), np.arange(-.5, 2.5, .01))
points_to_interp = [torch.from_numpy(
    X.flatten()).float(), torch.from_numpy(Y.flatten()).float()]

print([p.size() for p in points])
print([p.size() for p in points_to_interp])

fx = gi(points_to_interp)
print(fx)

fig, axes = plt.subplots(1, 2)

axes[0].imshow(np.sin(X) + 2 * np.cos(Y) + np.sin(5 * X * Y))
axes[0].set_title("True")
axes[1].imshow(fx.numpy().reshape(X.shape))
axes[1].set_title("Interpolated")
plt.savefig('interp.png')
"""

from mondrian.grid.utility import cell_centered_unit_grid
from mondrian.grid.grid_interpolator import RegularGridInterpolator

x = cell_centered_unit_grid((4,), device='cpu')
y = cell_centered_unit_grid((4,), device='cpu')
print(x)
values = torch.randn(4, 4, 3)
interp = RegularGridInterpolator(points=(x, y), values=values)

interp_size = 32

points_to_interp = cell_centered_unit_grid((interp_size, interp_size), device='cpu')
x = points_to_interp[0].flatten().contiguous()
y = points_to_interp[1].flatten().contiguous()
points_to_interp = [x, y]

interp_values = interp(points_to_interp)

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(values[..., 2])
axarr[1].imshow(interp_values.reshape(interp_size, interp_size, 3)[..., 2])
plt.savefig('interp.png')

"""
weights = torch.empty(2, 2, 4, 4, 4)
nn.init.kaiming_uniform_(weights)

up1 = interp(weights, size=(8, 8, 8))
up2 = interp(weights, size=(16, 16, 16))
up3 = interp(weights, size=(32, 32, 32))

fig, axarr = plt.subplots(2, 4)
axarr[0, 0].imshow(weights[0, 0, 0])
axarr[0, 1].imshow(up1[0, 0, 1])
axarr[0, 2].imshow(up2[0, 0, 1])
axarr[0, 3].imshow(up3[0, 0, 1])

axarr[1, 0].imshow(weights[0, 0, 1])
axarr[1, 1].imshow(up1[0, 0, 3])
axarr[1, 2].imshow(up2[0, 0, 3])
axarr[1, 3].imshow(up3[0, 0, 3])

plt.tight_layout()
plt.savefig('interp.png')
"""

"""
up1 = nn.functional.interpolate(weights, size=(8, 8), mode='bilinear', align_corners=False)
up2 = nn.functional.interpolate(weights, size=(16, 16), mode='bilinear', align_corners=False)
up3 = nn.functional.interpolate(weights, size=(32, 32), mode='bilinear', align_corners=False)

fig, axarr = plt.subplots(1, 4)
axarr[0].imshow(weights[0, 0])
axarr[1].imshow(up1[0, 0])
axarr[2].imshow(up2[0, 0])
axarr[3].imshow(up3[0, 0])

plt.tight_layout()
plt.savefig('interp.png')
"""


"""
weights = torch.empty(1, 2, 4, 4)
nn.init.kaiming_uniform_(weights)

grid1 = cell_centered_unit_grid((4, 4), device=weights.device).permute(2, 1, 0).unsqueeze(0)
grid2 = cell_centered_unit_grid((8, 8), device=weights.device).permute(2, 1, 0).unsqueeze(0)
grid3 = cell_centered_unit_grid((16, 16), device=weights.device).permute(2, 1, 0).unsqueeze(0)

grid1 = 2 * grid1 - 1
grid2 = 2 * grid2 - 1
grid3 = 2 * grid3 - 1

print(grid1.min(), grid1.max())
print(grid2.min(), grid2.max())
print(grid3.min(), grid3.max())


assert grid1.min() >= -1
assert grid1.min() <= 1
up1 = nn.functional.grid_sample(weights, grid1, mode='bilinear', align_corners=False)
up2 = nn.functional.grid_sample(weights, grid2, mode='bilinear', align_corners=False)
up3 = nn.functional.grid_sample(weights, grid3, mode='bilinear', align_corners=False)

fig, axarr = plt.subplots(1, 4)
axarr[0].imshow(weights[0, 0])
axarr[1].imshow(up1[0, 0])
axarr[2].imshow(up2[0, 0])
axarr[3].imshow(up3[0, 0])

plt.tight_layout()
plt.savefig('interp.png')
"""

"""
data = torch.randn(4, 4)
grid = CubicBSplineGrid2d.from_grid_data(data)

x = torch.linspace(0, 1, 100)
y = torch.linspace(0, 1, 100)
coords = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1).flatten(start_dim=0, end_dim=-2)
interps = grid(coords)
interps = interps.reshape(100, 100)


x = torch.linspace(0, 1, 50)
y = torch.linspace(0, 1, 50)
coords = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1).flatten(start_dim=0, end_dim=-2)
interps50 = grid(coords)
interps50 = interps50.reshape(50, 50)

x = torch.linspace(0, 1, 25)
y = torch.linspace(0, 1, 25)
coords = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1).flatten(start_dim=0, end_dim=-2)
interps25 = grid(coords)
interps25 = interps25.reshape(25, 25)

fig, axarr = plt.subplots(1, 4)
axarr[0].imshow(data.detach())
axarr[1].imshow(interps25.detach())
axarr[2].imshow(interps50.detach())
axarr[3].imshow(interps.detach())
plt.savefig('interp.png')

x = torch.rand(25, 25)
loss = (interps25 * x)
loss.sum().backward()


weights = torch.rand(5, 8, 8, 8, 8)
spline = CubicBSplineGrid4d.from_grid_data(weights)

w = torch.linspace(0, 1, 25)
x = torch.linspace(0, 1, 25)
y = torch.linspace(0, 1, 25)
z = torch.linspace(0, 1, 25)
coords = torch.stack(torch.meshgrid(w, x, y, z, indexing='ij'), dim=-1).flatten(start_dim=0, end_dim=-2)

data = spline(coords)
data = data.reshape(25, 25, 25, 25, 5)
print(data.size())

print(spline)
"""