from mondrian.grid.log_cpb import LogCPB

def test_log_cpb():
  lcpb = LogCPB(hidden_size=32, num_heads=4)
  bias = lcpb._get_grid(8, 8, 'cpu')
  assert True