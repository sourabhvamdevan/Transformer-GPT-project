

import torch

from config import *
from models.gpt import GPT
from utils.tokenizer import Tokenizer
from utils.dataset import Dataset
from utils.checkpoint import save

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(text)
dataset = Dataset(text, tokenizer)

model = GPT(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    xb, yb = dataset.get_batch("train")

    _, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % eval_interval == 0:
        print(f"Step {i} | Loss: {loss.item():.4f}")
        save(model, optimizer, i, "checkpoints/latest.pt")