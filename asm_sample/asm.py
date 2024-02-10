import torch
import numpy as np
import fdm_poisson
import matplotlib.pyplot as plt
from train import Solver
import math

from mondrian_lib.fdm_boundary_util import (
    BoundaryCondition,
    boundary_to_numpy,
    boundary_to_torch,
    read_boundary,
    write_boundary,
    boundary_to_vec
)

class SubdomainSolver:
    def __init__(self, xlim, ylim, res_per_unit):
        self.xlim = xlim
        self.ylim = ylim
        self.res_per_unit = res_per_unit
        self.m = int(xlim * res_per_unit)
        self.n = int(ylim * res_per_unit)

class FDMSubdomainSolver(SubdomainSolver):
    def __init__(self, xlim, ylim, res_per_unit):
        super().__init__(xlim, ylim, res_per_unit)
        self.f = np.ones((self.m-2, self.n-2))

    def solve(self, g: BoundaryCondition):
        # the fdm solver uses numpy, so have to convert from/to torch
        g_npy = boundary_to_numpy(g)        
        sol_npy = fdm_poisson.solve_poisson(g_npy, self.f, self.xlim, self.ylim)
        return torch.from_numpy(sol_npy)

class NNSubdomainSolver(SubdomainSolver):
    def __init__(self, model, xlim, ylim, res_per_unit):
        super().__init__(xlim, ylim, res_per_unit)
        self.model = model

    def solve(self, g: BoundaryCondition):
        input = boundary_to_vec(g)
        pred = self.model(input)
        return pred.reshape((self.n, self.m))

def asm(g: BoundaryCondition,
        xlim: float,
        ylim: float,
        subdomain_solver: SubdomainSolver):
    res_per_unit = subdomain_solver.res_per_unit
    global_m, global_n = int(xlim*res_per_unit), int(ylim*res_per_unit)
     
    is_bdy = torch.full((global_n, global_m), True)
    is_bdy[1:-1, 1:-1] = False

    u = torch.zeros((global_n, global_m))
    u = write_boundary(u, g)

    assert u.size(0) == global_n
    assert u.size(1) == global_m
    
    # place a subdomain every 0.8 units
    # the subdomain size should be a multiple
    subdomain_offset = 0.8
    n_subdomains = round(xlim / subdomain_offset)
    subdomain_step = round(res_per_unit * subdomain_offset)
    subdomain_start = torch.arange(0, u.size(1), subdomain_step).to(int)
    print(subdomain_start)

    iters = []
    for i in range(10):
        for subdomain in range(n_subdomains):
            start_col = subdomain_start[subdomain]
            subdomain = u[:,start_col:start_col + res_per_unit]
            assert subdomain.size(0) == subdomain_solver.n
            assert subdomain.size(1) == subdomain_solver.m

            g = read_boundary(subdomain)
            sol = subdomain_solver.solve(g).detach()
            write_boundary(sol, g)
            u[:,start_col:start_col + res_per_unit] = sol
        iters.append(u.clone())
    return iters, u

def series(x, res):
    total = torch.zeros(x.size(0))
    for k in range(1, res + 3):
        total += 2**-k * (torch.sin(k * 2 * math.pi * x) + 1)
    return total

def main():
    res_per_unit = 100
    ylim = 1
    xlim = 4.2
    width_res = int(xlim * res_per_unit)
    height_res = int(ylim * res_per_unit)

    x = torch.linspace(0, xlim, width_res)
    y = torch.linspace(0, ylim, height_res)
    top = series(x, width_res)

    top = torch.zeros(width_res)

    g = BoundaryCondition(
        top=top,
        right=torch.ones(height_res),
        bottom=torch.zeros(width_res),
        left=torch.ones(height_res) 
    )

    x, y = torch.meshgrid(x[1:-1], y[1:-1], indexing='xy')
    f = np.ones((height_res-2, width_res-2))

    g_npy = boundary_to_numpy(g)        
    gt_sol_npy = fdm_poisson.solve_poisson(g_npy, f, xlim, ylim)
    gt_sol = torch.from_numpy(gt_sol_npy)

    fdm_ss = FDMSubdomainSolver(1, 1, res_per_unit)
    asm_iters, fdm_sol = asm(g, xlim, ylim, fdm_ss)

    #model = torch.load('model.pt')
    #nn_ss = NNSubdomainSolver(model, 1, 1, res_per_unit)
    #asm_iters, nn_sol = asm(g, xlim, ylim, nn_ss)

    for idx, asm_iter in enumerate(asm_iters):
        fig, axarr = plt.subplots(2, 1)
        axarr[0].imshow(gt_sol.cpu())
        axarr[1].imshow(asm_iter.detach().cpu())
        plt.savefig(f'asm_sol{idx}.png')
        plt.close()

    # mean of fft of u in the y-direction
    gt_h_y = torch.fft.rfft(gt_sol[:, 160:160+res_per_unit], dim=0).abs()
    gt_h_y_mean = gt_h_y.mean(dim=1)
    plt.plot(gt_h_y_mean, label='ground truth', c='b')

    for idx, i in enumerate(asm_iters):
        nn_h_y = torch.fft.rfft(i[:, 160:160+res_per_unit], dim=0).abs()
        nn_h_y_mean = nn_h_y.mean(dim=1)
        plt.plot(nn_h_y_mean, label=f'nn-iter {idx}')
        plt.yscale('log')
    plt.legend()
    plt.savefig('spectra.png')

if __name__ == '__main__':
    main()
