import torch
from torch import nn
import torch.nn.functional as F

class LearnedPosEmbedding2d(nn.Module):
    r"""
    This is a simple learned position embedding, meant for fixed size sequences.
    """
    def __init__(self, seq_len, channels):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        
        # This is based on timm's vit position embedding, which hardcodes N(0, 0.02).
        # Torch's nn.Embedding initialize to a N(0, 1), which seems too large.
        # This only uses channels, and does not include height and width. I found
        # interpolating the height and width completely broke the model when trying to apply
        # to higher resolutions.
        self.pos_embed = nn.Parameter(torch.randn(seq_len, self.channels, 1, 1) * .02)

    def forward(self, f: torch.Tensor):
        seq_len = f.size(1)
        func_height = f.size(-2)
        func_width = f.size(-1)

        assert seq_len == self.seq_len

        cont_pos_embedding = self.pos_embed.expand(-1, -1, func_height, func_width)
                
        return f + cont_pos_embedding