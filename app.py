

import streamlit as st
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
model.eval()

st.title("GPT From Scratch")
st.write("Character Level GPT")

prompt = st.text_input("Enter Prompt")

if st.button("Generate"):

    if prompt:
        prompt = "".join(c for c in prompt if c in tokenizer.stoi)
        if prompt:
            context = torch.tensor(
                [tokenizer.encode(prompt)],
                dtype=torch.long,
                device=device
            )
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    output = model.generate(
        context,
        max_new_tokens=150,
        temperature=0.8
    )[0].tolist()

    st.text(tokenizer.decode(output))