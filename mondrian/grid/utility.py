

def is_power_of_2(x: int):
  assert isinstance(x, int)
  # gross way of checking number of 1 bits...,
  return bin(x).count('1') == 1