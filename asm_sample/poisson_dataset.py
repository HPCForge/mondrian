import fdm_poisson
import numpy as np
import h5py
import matplotlib.pyplot as plt


class BCGenerator:
    def __init__(self):
        self.rng = np.random.default_rng()

    def get_edge_type(self):
        t = self.rng.integers(0, 10)
        return t

    def get_edge(self, n):
        edge_type = self.get_edge_type()
        if edge_type == 0:
            return np.zeros(n)
        if edge_type < 3:
            val = self.rng.uniform(-1, 1)
            return np.full(n, val)
        if edge_type < 11:
            amp = self.rng.uniform(low=-10, high=10)
            freq = self.rng.uniform(low=1, high=20)
            inner_offset = self.rng.uniform(0, 2 * np.pi)
            outer_offset = self.rng.uniform(-1, 1)
            x = np.linspace(0, 1, n)
            return amp * np.sin(freq * x + inner_offset) + outer_offset

    def get_bc(self, m, n):
        return fdm_poisson.BoundaryCondition(
            top=self.get_edge(n),
            right=self.get_edge(m),
            bottom=self.get_edge(n),
            left=self.get_edge(m),
        )

def main():
    gen = BCGenerator()

    xlim, ylim = 1, 1
    res_per_unit = 100
    m, n = xlim*res_per_unit, ylim*res_per_unit
    f = np.ones((m-2,n-2))

    with h5py.File('./poisson.hdf5', 'w') as handle:
        handle.attrs['xlim'] = xlim
        handle.attrs['ylim'] = ylim
        handle.attrs['res_per_unit'] = res_per_unit
        handle.attrs['m'] = m
        handle.attrs['n'] = n

        num_samples = 10000
        zfill_cnt = len(str(num_samples))
        for idx in range(num_samples):
            if idx % 10 == 0:
                print(f'[{idx}/{num_samples}]')
            poisson_group = handle.create_group(f'{idx}'.zfill(zfill_cnt))

            g = gen.get_bc(m, n)
            u = fdm_poisson.solve_poisson(g, f, xlim, ylim)

            poisson_group.create_dataset('sol', data=u)

    plt.imshow(u)
    plt.savefig('poisson_sol.png')

if __name__ == '__main__':
    main()
