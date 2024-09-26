

class DDOperator(nn.Module):
    def __init__(self, in_channels, out_channels):

        hidden_channels = 128
        self.kernel = nn.Sequential(
                nn.Linear(2, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 1))

    def forward(self,
                v,
                src_coords,
                tgt_coords,
                adj_t):
        r"""
        Args:
            v: [J, in_channels]
            src_coords: [J, 2]
            tgt_coords: [J', 2]
            adj_t: [J, J], a sparse adjacency matrix capturing subdomain pattern.
        Returns:
            u: [J', out_channels]
        """
        v + adj_t.matmul(v)
        
