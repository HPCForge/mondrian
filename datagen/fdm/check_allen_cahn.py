import h5py

with h5py.File('./allen_cahn.hdf5') as f:
    group1 = f['2_2']
    for sim in group1.keys():
        sol = group1[sim]['solution'][:]
        print(sol.min(), sol.max(), sol.mean())
        print(group1[sim].attrs['xlim'])
        print(group1[sim].attrs['ylim'])
        # Allen-Cahn satisfies a maximum principle.
        # if |u(x, 0)| <= 1, then |u(x, t)| <= 1 
        assert(abs(sol).max() <= 1)
