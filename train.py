import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import tokenize, stem, stem_list, bag_of_words, ignore_symbols

from model import NeuralNet

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
num_of_workers = 0

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 10000


dataset = ChatDatset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_of_workers)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training loop
for epoch in range(num_epochs):
    for (w, label) in train_loader:
        w = w.to(device)
        label = label.to(dtype=torch.long).to(device)

        # forward
        output = model(w)
        loss = criterion(output, label)

        # backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1} / {num_epochs}, Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "all_tags": tags
}

FILE = "data.pth"

torch.save(data, FILE)

print(f"Training complete, file saved to {FILE}")