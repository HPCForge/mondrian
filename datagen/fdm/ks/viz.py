import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("./ks_3_32.hdf5") as f:
    for size_key in f.keys():
        group1 = f[size_key]
        for idx, sim in enumerate(group1.keys()):
            if idx >= 10:
                break
            sol = group1[sim]["solution"][:]
            init = sol[0]
            target = sol[-1]
            print(target.min(), target.max())
            
            fig, axarr = plt.subplots(nrows=1, ncols=2)
            
            init_fft = np.fft.fft2(init)
            target_fft = np.fft.fft2(target)
            
            axarr[0].imshow(init)
            axarr[1].imshow(target)
            plt.savefig(f"ac{idx}.png")
            plt.close()