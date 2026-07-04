

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from models.block import Block

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size, n_embd)
        self.pos_emb=nn.Embedding(block_size, n_embd)
        self.blocks=nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln=nn.LayerNorm(n_embd)
        self.head=nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T=idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.blocks(x)
        logits=self.head(self.ln(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx