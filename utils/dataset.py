

import torch
from config import batch_size, block_size, device

class Dataset:
    def __init__(self, text, tokenizer):
        data=torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data=data[:n]
        self.val_data=data[n:]

    def get_batch(self, split):
        data=self.train_data if split=="train" else self.val_data
        ix=torch.randint(len(data) - block_size, (batch_size,))
        x=torch.stack([data[i:i + block_size] for i in ix])
        y=torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)