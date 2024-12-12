import torch
from torch import nn
import torch.nn.functional as F


class FuncPosEmbedding2d(nn.Module):
    def __init__(self, max_seq_len, channels):
        super().__init__()
        self.h = 4
        self.w = 4
        self.channels = channels
        self.embedding = nn.Embedding(
            max_seq_len, embedding_dim=self.channels * self.h * self.w
        )

    def forward(self, f: torch.Tensor):
        seq_len = f.size(1)
        func_height = f.size(-2)
        func_width = f.size(-1)

        seq_idx = torch.arange(seq_len, device=self.embedding.weight.device)
        disc_pos_embedding = self.embedding(seq_idx).reshape(
            (-1, self.channels, self.h, self.w)
        )
        # [seq_len x channels x height x width]
        cont_pos_embedding = F.interpolate(
            disc_pos_embedding, (func_height, func_width), mode="bilinear"
        )
        return f + cont_pos_embedding.unsqueeze(0)
