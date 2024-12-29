import torch
from torch import nn
import einops


class VectorSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        r"""
        Args:
          x: [batch, seq, embed]
        Returns:
          [batch, seq, embed]
        """
        # passing one input for query, key, value will do self-attention.
        # Setting need_weights=False should enable using mha's fast path.
        x, _ = self.mha(x, x, x, need_weights=False)
        return x
