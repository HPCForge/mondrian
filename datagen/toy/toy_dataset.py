import torch
import matplotlib.pyplot as plt
import h5py

def periodic_1d(x, xamp, freq):
  x = x.unsqueeze(-1) * freq
  xamp = xamp.unsqueeze(0)
  return (xamp * torch.sin(x)).sum(-1)

def periodic_2d(x, y, xamp, yamp, freq):
  x = x.unsqueeze(-1) * freq
  y = y.unsqueeze(-1) * freq
  xamp = xamp.unsqueeze(0)
  yamp = yamp.unsqueeze(0)
  return (torch.sin(x) * torch.sin(y)).sum(-1)

"""
# cell-centered coords on [0, 2pi]
num_points = 64
delta = (2 * torch.pi) / (num_points + 1)
x = (torch.arange(num_points) + 0.5) * delta

num_freq = 20
freq = torch.arange(num_freq) + 1
xamp = torch.rand(num_freq) + 0.01
p = periodic_1d(x, xamp, freq)

operator_weights = torch.rand(num_points // 2 + 1) + 0.1

def operator(p, weights):
  p_h = torch.fft.rfft(p)
  data = torch.fft.irfft(p_h * weights)
  data = torch.nn.functional.tanh(data)
  data = torch.fft.rfft(data)
  data = torch.fft.irfft(data * weights)
  return data

data = operator(p, operator_weights)

fig, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(p)
axarr[0, 1].plot((abs(torch.fft.fft(p)))[:32])
axarr[1, 0].plot(data)
axarr[1, 1].plot((abs(torch.fft.fft(data)))[:32])

axarr[0, 0].set_title('input')
axarr[0, 1].set_title('input fft')
axarr[1, 0].set_title('output')
axarr[1, 1].set_title('output fft')

plt.tight_layout()
plt.savefig('periodic.png')
"""

def operator(p, weights):
  r"""
  This is just a simple non-linear operator that 
  scales different amplitudes randomly
  """
  p_h = torch.fft.rfft(p)
  data = torch.fft.irfft(p_h * weights)
  data = torch.nn.functional.tanh(data)
  data = torch.fft.rfft(data)
  data = torch.fft.irfft(data * weights)
  return data

def build_dataset(size, num_points, filename):
  # cell-centered coords on [0, 2pi]
  delta = (2 * torch.pi) / (num_points + 1)
  x = (torch.arange(num_points) + 0.5) * delta

  num_freq = 20
  dataset_input = []
  dataset_label = []
  for idx in range(size):
    freq = torch.arange(num_freq) + 1
    xamp = torch.rand(num_freq) + 0.01
    p = periodic_1d(x, xamp, freq)
    operator_weights = torch.rand(num_points // 2 + 1) + 0.1
    l = operator(p, operator_weights)

    dataset_input.append(p)
    dataset_label.append(l)
    
  dataset_input = torch.stack(dataset_input)
  dataset_label = torch.stack(dataset_label)

  with h5py.File(filename, 'w') as handle:
    handle.create_dataset('input', data=dataset_input)
    handle.create_dataset('label', data=dataset_label)

train_size = 10000
val_size = 1000
test_size = 1000

build_dataset(train_size, 64, 'train_64.hdf5')
build_dataset(val_size, 64, 'val_64.hdf5')
build_dataset(test_size, 32, 'test_32.hdf5')
build_dataset(test_size, 96, 'test_64.hdf5')
build_dataset(test_size, 128, 'test_128.hdf5')

