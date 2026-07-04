

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import n_embd, n_head, block_size, dropout

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embd, head_size, bias=False)
        self.query=nn.Linear(n_embd, head_size, bias=False)
        self.value=nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        w = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = self.dropout(F.softmax(w, dim=-1))
        return w @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hs = n_embd // n_head
        self.heads=nn.ModuleList([Head(hs) for _ in range(n_head)])
        self.proj=nn.Linear(n_embd, n_embd)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(x))