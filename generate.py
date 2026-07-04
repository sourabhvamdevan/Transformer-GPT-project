

import os
import torch

from config import *
from models.gpt import GPT
from utils.tokenizer import Tokenizer
from utils.checkpoint import load

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(text)

model = GPT(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

load(model, optimizer, "checkpoints/latest.pt", device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)

output = model.generate(
    context,
    max_new_tokens=500,
    temperature=0.8
)[0].tolist()

generated = tokenizer.decode(output)

print(generated)

os.makedirs("outputs", exist_ok=True)

with open("outputs/generated.txt", "w", encoding="utf-8") as f:
    f.write(generated)