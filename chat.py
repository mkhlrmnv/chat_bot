import random
import json
from datetime import datetime

import torch

from model import NeuralNet
from utils import bag_of_words, tokenize

import requests

api_key = '5644d22f8412fa3d962a30d9a3cb6b3d'
home_lat = '60.155926'
home_long = '24.910053'
base_url = f'https://api.openweathermap.org/data/2.5/weather?lat={home_lat}&lon={home_long}&appid={api_key}'

# TODO Move to switch case
response = requests.get(base_url)
data = response.json()

print(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['all_tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Lil_Mixu"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                match tag:
                    case "time":
                        print(f"{bot_name}: Time is: {random.choice(intent['responses'])}"
                              f"{datetime.now().strftime('%H:%M:%S')}")
                    case "weather":
                        pass
                        # print(f"{bot_name}: {random.choice(intent['responses'])}{getweather()}")
                    case "schedule":
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                    case _:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")