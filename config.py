import torch

batch_size = 4
block_size = 16

max_iters = 300
eval_interval = 50
eval_iters = 10

learning_rate = 3e-4

n_embd = 32
n_head = 2
n_layer = 1

dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"