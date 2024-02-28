import torch

def cell_centered_meshgrid(xlim, ylim, x_res, y_res):
    # FDM data uses a cell-centered stencil,
    # so the true coordinates are offset by 0.5
    # we remove the last coordinate, since it is outside the domain
    x_coords, y_coords = torch.meshgrid(
            0.5 / x_res + torch.linspace(0, xlim, x_res + 1)[:-1],
            0.5 / y_res + torch.linspace(0, ylim, y_res + 1)[:-1],
            indexing='xy')
    return x_coords, y_coords
