import torch
import numpy as np
import fdm_poisson
from fdm_poisson import BoundaryCondition
import matplotlib.pyplot as plt
from train import Solver

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
        g_npy = fdm_poisson.boundary_to_numpy(g)        
        sol_npy = fdm_poisson.solve_poisson(g_npy, self.f, self.xlim, self.ylim)
        return torch.from_numpy(sol_npy)

class NNSubdomainSolver(SubdomainSolver):
    def __init__(self, model, xlim, ylim, res_per_unit):
        super().__init__(xlim, ylim, res_per_unit)
        self.model = model

    def solve(self, g: BoundaryCondition):
        input = fdm_poisson.boundary_to_vec(g)
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
    u = fdm_poisson.write_boundary(u, g)

    assert u.size(0) == global_n
    assert u.size(1) == global_m
    
    # place a subdomain every 0.8 units
    # the subdomain size should be a multiple
    subdomain_offset = 0.8
    n_subdomains = round(xlim / subdomain_offset)
    subdomain_step = round(res_per_unit * subdomain_offset)
    subdomain_start = torch.arange(0, u.size(1), subdomain_step).to(int)
    print(subdomain_start)

    for i in range(10):
        for subdomain in range(n_subdomains):
            start_col = subdomain_start[subdomain]
            subdomain = u[:,start_col:start_col + res_per_unit]
            assert subdomain.size(0) == subdomain_solver.n
            assert subdomain.size(1) == subdomain_solver.m

            g = fdm_poisson.read_boundary(subdomain)
            sol = subdomain_solver.solve(g)
            fdm_poisson.write_boundary(sol, g)
            u[:,start_col:start_col + res_per_unit] = sol

        print(u)
        plt.imshow(u.detach().cpu())
        plt.savefig(f'asm_sol{i}.png')

def main():
    res_per_unit = 100
    ylim = 1
    xlim = 2.6
    width_res = int(xlim * res_per_unit)
    height_res = int(ylim * res_per_unit)

    g = BoundaryCondition(
        top=torch.ones(width_res) / 2,
        right=torch.ones(height_res) / 2,
        bottom=torch.zeros(width_res),
        left=torch.ones(height_res) / 2,
    )

    #fdm_ss = FDMSubdomainSolver(1, 1, res_per_unit)
    #asm(g, xlim, ylim, fdm_ss)

    model = torch.load('model.pt')
    nn_ss = NNSubdomainSolver(model, 1, 1, res_per_unit)
    asm(g, xlim, ylim, nn_ss)

if __name__ == '__main__':
    main()
