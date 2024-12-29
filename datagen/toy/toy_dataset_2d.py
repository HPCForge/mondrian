import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
# This is from pylians: https://pylians3.readthedocs.io/en/master/gaussian_fields.html
import density_field_library as DFL
from scipy.interpolate import RegularGridInterpolator
  
def gaussian_field_2d(grid_res, box_size, k, Pk, seed):
    """Use Pylians to generate a Gaussian random field.
    Args:
        grid_res: number of grid cells
        k: 1D array of frequency
        Pk: 1D array of magnitudes to interpolate. The problem should
            get "easier" as Pk goes to zero.
        seed: Random seed
    Returns:
        A numpy array containing the field
    """
    assert torch.all(k >= 0)
    df_3d = torch.from_numpy(
      DFL.gaussian_field_2D(grid_res, k.numpy(), Pk.numpy(), 0, seed, box_size, threads=1)
    )
    assert torch.isclose(df_3d.mean(), torch.tensor(0, dtype=torch.float32))
    return df_3d

def operator(p, weights1, weights2):
  p_h = torch.fft.rfft2(p)
  data = torch.fft.irfft2(p_h * weights1)
  data = torch.nn.functional.gelu(data)
  data = torch.fft.rfft2(data)
  data = torch.fft.irfft2(data * weights2)
  return data

"""
num_points = 128
box_size = 2 * torch.pi
k = torch.arange(128, dtype=torch.float32) + 1
Pk = k ** -4

p = gaussian_field_2d(num_points, box_size, k, Pk, 0)

weights1 = 2 * torch.rand(num_points, num_points // 2 + 1) + 0.1
weights2 = 2 * torch.rand(num_points, num_points // 2 + 1) + 0.1

data = operator(p, weights1, weights2)

fig, axarr = plt.subplots(2, 3)
axarr[0, 0].imshow(p, cmap='seismic')
axarr[0, 1].imshow(p, vmin=-0.0001, vmax=0.0001)
axarr[0, 2].imshow(torch.fft.fftshift(torch.log(abs(torch.fft.fft2(p))), dim=(-2, -1)), vmin=0)

axarr[1, 0].imshow(data, cmap='seismic')
axarr[1, 1].imshow(data, vmin=-0.0001, vmax=0.0001)
axarr[1, 2].imshow(torch.fft.fftshift(torch.log(abs(torch.fft.fft2(data))), dim=(-2, -1)), vmin=0)

axarr[0, 0].set_title('input')
axarr[0, 1].set_title('input zerba')
axarr[0, 2].set_title('input log fft')
axarr[1, 0].set_title('output')
axarr[1, 1].set_title('output zebra')
axarr[1, 2].set_title('output log fft')

plt.tight_layout()
plt.savefig('periodic.png')
"""


def cell_centered_grid(num_points):
  # cell-centered coords on [0, 2pi]
  delta = (2 * torch.pi) / (num_points + 1)
  x = (torch.arange(num_points) + 0.5) * delta
  return x.numpy()

def cell_centered_grid_2d(num_points):
  return np.stack(
    np.meshgrid(
      cell_centered_grid(num_points),
      cell_centered_grid(num_points),
      indexing='xy'
    ),
    axis=-1
  )

def build_dataset(size, num_interp_points, filename):
  r"""
  Generates a dataset by computing a gaussian random field on
  128 points, applying some stupid operator to it, and then
  interpolating the output.
  """
  # settings for the GRF
  num_points = 128
  box_size = 2 * torch.pi
  k = torch.arange(128, dtype=torch.float32) + 1
  Pk = k ** -4

  # random weights are used for the operator
  weights1 = 2 * torch.rand(num_points, num_points // 2 + 1) + 0.1
  weights2 = 2 * torch.rand(num_points, num_points // 2 + 1) + 0.1
  
  dataset_input = []
  dataset_label = []
  
  for idx in range(size):
    # Generate a GRF and apply operator to it
    grf = gaussian_field_2d(num_points, box_size, k, Pk, idx)
    weights1 = torch.rand(num_points // 2 + 1) + 0.1
    weights2 = torch.rand(num_points // 2 + 1) + 0.1
    label = operator(grf, weights1, weights2)
    
    # use scipy to interpolate on desired cell-centered grid
    grf = grf.numpy()
    label = label.numpy()
    
    # get points of original grid
    points = cell_centered_grid(num_points)
    
    # get points for interpolated grid
    interp_points = cell_centered_grid_2d(num_interp_points)
    interp_points = interp_points.reshape(-1, 2)

    input_interp = grf
    label_interp = label
    
    # If the desired grid is different, we interpolate it
    if num_points != num_interp_points:
      # interpolate the input
      input_interpolator = RegularGridInterpolator((points, points), grf, method='cubic')
      input_interp = input_interpolator(interp_points)
      input_interp = input_interp.reshape(num_interp_points, num_interp_points)
      
      # interpolate the label
      label_interpolator = RegularGridInterpolator((points, points), label, method='cubic')
      label_interp = label_interpolator(interp_points)
      label_interp = label_interp.reshape(num_interp_points, num_interp_points)

    dataset_input.append(torch.from_numpy(input_interp))
    dataset_label.append(torch.from_numpy(label_interp))
      
  dataset_input = torch.stack(dataset_input).to(torch.float32)
  dataset_label = torch.stack(dataset_label).to(torch.float32)

  with h5py.File(filename, 'w') as handle:
    handle.create_dataset('input', data=dataset_input)
    handle.create_dataset('label', data=dataset_label)

train_size = 10000
val_size = 1000
test_size = 1000

build_dataset(train_size, 64, 'train_64.hdf5')
build_dataset(val_size, 64, 'val_64.hdf5')

build_dataset(test_size, 16, 'test_16.hdf5')
build_dataset(test_size, 32, 'test_32.hdf5')
build_dataset(test_size, 48, 'test_48.hdf5')
build_dataset(test_size, 64, 'test_64.hdf5')
build_dataset(test_size, 80, 'test_80.hdf5')
build_dataset(test_size, 96, 'test_96.hdf5')
build_dataset(test_size, 112, 'test_112.hdf5')

with h5py.File('test_64.hdf5') as handle:
  input_64 = handle['input'][0]
  label_64 = handle['label'][0]

with h5py.File('test_96.hdf5') as handle:
  input_128 = handle['input'][0]
  label_128 = handle['label'][0]

fig, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(input_64, cmap='seismic')
axarr[0, 1].imshow(label_64, cmap='seismic')

axarr[1, 0].imshow(input_128, cmap='seismic')
axarr[1, 1].imshow(label_128, cmap='seismic')

axarr[0, 0].set_title('input')
axarr[0, 1].set_title('label')
axarr[1, 0].set_title('input')
axarr[1, 1].set_title('label')
plt.savefig('periodic.png')