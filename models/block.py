

import torch.nn as nn
from models.attention import MultiHeadAttention
from models.feedforward import FeedForward
from config import n_embd

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=MultiHeadAttention()
        self.ff=FeedForward()
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x