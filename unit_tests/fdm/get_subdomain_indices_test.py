import torch
from mondrian_lib.fdm.get_subdomain_indices import get_subdomain_indices

def test_get_subdomain_indices_1():
    xlim = 2
    ylim = 3
    x_res, y_res = xlim * 20, ylim * 20
    x_idx, y_idx, res_per_x, res_per_y = get_subdomain_indices(
            0.2,
            x_res,
            y_res,
            subdomain_size_x=0.5,
            subdomain_size_y=0.5,
            domain_size_x=xlim,
            domain_size_y=ylim)

    # the subdomains should overlap
    for i in range(2):
        assert x_idx[i] + res_per_x > x_idx[i + 1]
    for i in range(2):
        assert y_idx[i] + res_per_y > y_idx[i + 1]

    assert x_idx[-1] + res_per_x == x_res
    assert y_idx[-1] + res_per_y == y_res
