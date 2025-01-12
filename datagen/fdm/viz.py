import h5py
import matplotlib.pyplot as plt

with h5py.File("./fix_allen_cahn_20000_32.hdf5") as f:
    for size_key in f.keys():
        group1 = f[size_key]
        for idx, sim in enumerate(group1.keys()):
            if idx >= 10:
                break
            sol = group1[sim]["solution"][:]
            init = sol[0]
            target = sol[1]
            print(target.min(), target.max())
            fig, axarr = plt.subplots(nrows=1, ncols=2)
            axarr[0].imshow(init, cmap="plasma", vmin=-1, vmax=1)
            axarr[1].imshow(target, cmap="plasma", vmin=-1, vmax=1)
            plt.savefig(f"ac{idx}.png")
