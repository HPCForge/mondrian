import h5py
import matplotlib.pyplot as plt

with h5py.File('./allen_cahn.hdf5') as f:
    for size_key in f.keys():
        group1 = f[size_key]
        for idx, sim in enumerate(group1.keys()):
            if idx >= 10: break
            sol = group1[sim]['solution'][:]
            init = sol[0]
            target = sol[1]
            fig, axarr = plt.subplots(nrows=1, ncols=2)
            axarr[0].imshow(init, cmap='plasma')
            axarr[1].imshow(target, cmap='plasma')
            plt.savefig(f'ac{idx}.png')