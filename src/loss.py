import torch
from torch.nn.functional import mse_loss

def rep_mse(output, actual):
    return ((output - actual) ** 2).mean(dim=(1, 2, 3)).mean()

def rel_mse(output, actual):
    num = (((output - actual) / (actual)) ** 2).mean(dim=(1, 2, 3)).mean()
    return num
   
input = torch.randn(5, 32, 64, 64)
label = input + 0.001 # torch.randn(5, 32, 64, 64)

print(mse_loss(input, label))
print(rep_mse(input, label))

print(rel_mse(input, label))