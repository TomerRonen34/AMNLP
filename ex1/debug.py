import torch
import sys
from os.path import dirname

sys.path.append('.')
sys.path.append('..')
sys.path.append('/content/drive/MyDrive/TAU/Advanced NLP/Ex1')

import data_loader
from traineval import train, evaluate
import model as model

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"deviced used is {device}")

import importlib

importlib.reload(model)

import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed = 42
set_seed(seed)

train_dataset, tokens_vocab, y_vocab = data_loader.load_train_dataset()
dev_dataset = data_loader.load_dev_dataset(tokens_vocab, y_vocab)

sa_train_dataset = data_loader.WSDSentencesDataset.from_word_dataset(train_dataset)
sa_dev_dataset = data_loader.WSDSentencesDataset.from_word_dataset(dev_dataset)

lr = 2e-4
dropout = 0.2
D = 300
batch_size = 20
num_epochs = 2
set_seed(seed)

m = model.WSDModel(
    tokens_vocab.size(),
    y_vocab.size(),
    D=D,
    dropout_prob=dropout,
    use_padding=True,
    use_positional_encodings=True,
).to(device)

optimizer = torch.optim.Adam(m.parameters(), lr=lr)

losses, train_acc, val_acc = train(
    m, optimizer, sa_train_dataset, sa_dev_dataset, num_epochs=num_epochs, batch_size=batch_size)
