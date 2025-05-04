import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("./fix_grid_allen_cahn_10_128.hdf5") as f:
    for size_key in f.keys():
        group1 = f[size_key]
        for idx, sim in enumerate(group1.keys()):
            if idx >= 10:
                break
            sol = group1[sim]["solution"][:]
            init = sol[0]
            target = sol[1]
            print(target.min(), target.max())
            fig, axarr = plt.subplots(nrows=1, ncols=5)
            
            init_fft = np.fft.fft2(init)
            target_fft = np.fft.fft2(target)
            
            axarr[0].imshow(init, cmap="plasma", vmin=-1, vmax=1)
            axarr[1].imshow(np.fft.fftshift(np.log(np.abs(init_fft))))
            axarr[2].imshow(target, cmap="plasma", vmin=-1, vmax=1)
            axarr[3].imshow(np.fft.fftshift(np.log(np.abs(target_fft))))
            plt.savefig(f"ac{idx}.png")
            plt.close()
            
            init_mean_x = init_fft.mean(axis=-1)
            init_mean_y = init_fft.mean(axis=-2)
            
            plt.plot(abs(init_mean_x)[1:64], label='mean x-amplitude')
            plt.plot(abs(init_mean_y)[1:64], label='mean y-amplitude')
            plt.legend()
            plt.yscale('log')
            plt.savefig('freq.png')
            plt.close()