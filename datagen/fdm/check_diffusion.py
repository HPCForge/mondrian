import h5py

with h5py.File('./diffusion.hdf5', 'a') as f:
    group1 = f['1_1']
    for sim in group1.keys():
        diff = group1[sim]['diffusivity'][:]
        sol = group1[sim]['solution'][:]
        print(diff.min(), diff.max(), diff.mean())
        print(sol.min(), sol.max(), sol.mean())
        print(group1[sim].attrs['xlim'])
        print(group1[sim].attrs['ylim'])
        # by the maximum principle, the solution should
        # vary between 0 and the maximum of the initial condition.
        # Since the initial condition is uniform random between [0.1,0.5]
        # the solution should be in the range [0, 0.5].
        # if this is violated, we just remove it from the dataset
        if not (sol.min() >= 0 and sol.max() <= 0.50):
            del group1[sim]
    print(f'remaining sims = {len(group1.keys())}')
        #assert(sol.min() >= 0 and sol.max() <= 0.51)
