import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
  
'''
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
'''

def cell_centered_grid(num_points):
  # cell-centered coords on [0, 1.5]
  delta = 1 / (num_points)
  x = (torch.arange(num_points) + 0.5) * delta
  x = 1.5 * x
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
  num_points = num_interp_points

  rng = np.random.default_rng()
  
  dataset_input = []
  dataset_label = []
  
  for idx in range(size):
    a = rng.uniform(low=1, high=1.2)
    a = np.full((num_points, num_points), a)

    grid_points = cell_centered_grid_2d(num_points)
    x, y = grid_points[..., 0], grid_points[..., 1]
    
    input = np.exp(a * x * y)
    label = a * y * np.exp(a * x * y)
    """
    # get points of original grid
    points = cell_centered_grid(num_points)
    
    # get points for interpolated grid
    interp_points = cell_centered_grid_2d(num_interp_points)
    interp_points = interp_points.reshape(-1, 2)

    input_interp = input
    label_interp = label
    
    # If the desired grid is different, we interpolate it
    if num_points != num_interp_points:
      # interpolate the input
      input_interpolator = RegularGridInterpolator((points, points), input, method='cubic')
      input_interp = input_interpolator(interp_points)
      input_interp = input_interp.reshape(num_interp_points, num_interp_points)
      
      # interpolate the label
      label_interpolator = RegularGridInterpolator((points, points), label, method='cubic')
      label_interp = label_interpolator(interp_points)
      label_interp = label_interp.reshape(num_interp_points, num_interp_points)
    """
    dataset_input.append(torch.from_numpy(input))
    dataset_label.append(torch.from_numpy(label))

  dataset_input = torch.stack(dataset_input).to(torch.float32)
  dataset_label = torch.stack(dataset_label).to(torch.float32)

  with h5py.File(filename, 'w') as handle:
    handle.create_dataset('input', data=dataset_input)
    handle.create_dataset('label', data=dataset_label)

train_size = 2000
val_size = 1000
test_size = 1000
fine_size = 100


build_dataset(train_size, 16, 'train_16.hdf5')
build_dataset(val_size, 16, 'val_16.hdf5')

build_dataset(train_size, 64, 'train_64.hdf5')
build_dataset(val_size, 64, 'val_64.hdf5')

for res in [16, 32, 48, 64, 80, 96, 112, 128, 256, 512]:
  build_dataset(test_size, res, f'test_{res}.hdf5')
  build_dataset(fine_size, res, f'fine_{res}.hdf5')

print('done')

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
