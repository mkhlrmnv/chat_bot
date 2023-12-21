import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import tokenize, stem, stem_list, bag_of_words, ignore_symbols

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


all_words = stem_list(all_words)
all_words = ignore_symbols(stem_list(all_words))
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(all_words)


x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDatset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples


# parameters
batch_size = 8
num_of_workers = 2

dataset = ChatDatset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_of_workers)