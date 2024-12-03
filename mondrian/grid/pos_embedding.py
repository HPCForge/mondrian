import torch
from torch import nn
import torch.nn.functional as F

class FuncPosEmbedding2d(nn.Module):
  def __init__(self, max_seq_len, channels):
    super().__init__()
    self.h = 8
    self.w = 8
    self.channels = channels
    self.embedding = nn.Embedding(max_seq_len, embedding_dim=self.channels * self.h * self.w)
    
  def forward(self, seq_len: int, func_height: int, func_width: int):
    seq_idx = torch.arange(seq_len, device=self.embedding.weight.device)
    disc_pos_embedding = self.embedding(seq_idx).reshape((-1, self.channels, self.h, self.w))
    # [seq_len x channels x height x width]
    cont_pos_embedding = F.interpolate(disc_pos_embedding, 
                                       (func_height, func_width), 
                                       mode='bicubic')
    # [1 x seq_len x channels x height x width]
    return cont_pos_embedding.unsqueeze(0)