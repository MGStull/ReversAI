import Transformer
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm,trange
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 64
trg_vocab_size = 3
model = Transformer(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx).to(device)
out = model(x,trg[:,:-1])
print(out.shape)

optimizer = torch.nn.optim.Adam()
train_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_train.csv"
train_dataset = OthelloDataset(train_csv_path, token_to_idx)
test_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_test.csv"
test_dataset = OthelloDataset(test_csv_path, token_to_idx)
# Create DataLoader with custom collate function
batch_size = 32
dataloader_train = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
dataloader_test = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

