#!/bin/python

import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
import random
from tqdm.autonotebook import tqdm


# Generelle Initialisierung
logging.basicConfig(filename="sample2.log", level=logging.DEBUG)

## Check installed device for calculations
if torch.has_cuda:
    device = "cuda:0"
elif torch.has_mps:
    device = "mps"
else:
    device = "cpu"
print(f"processing device is: {device}")



## Definition Feed-Forward netz mit einem hidden LAyer
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.hidden = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        startFeedForward = timer()
        y_1 = self.input_layer(x)
        y_non_linear = self.sigmoid(y_1)
        out = self.hidden(y_non_linear)
        logging.debug("FeedForward.forward() %f ms" % ((timer()-startFeedForward)*1000.0))

        return out


## Implementierung der Data-Klasse
from typing import Dict, List, Set, Tuple
from torch.utils.data import Dataset

class LanguagesDataset(Dataset):
    def __init__(self, rows) -> None:
        self.rows: List[Dict[str, str]] = [row for row in rows]

    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx) -> Tuple[str, str]:
        row = self.rows[idx]
        return row["labels"], row["text"]




## Hauptprogramm

print("Language Classification with PyTorch")

# Load data
TRAIN_FILE = "data/test.json"
print("loading %s ..." % (TRAIN_FILE))

with open(TRAIN_FILE, "rt", encoding="utf8") as train_file:
    _training = json.load(train_file)
print(f"loading done: loaded {len(_training)} rows.")
training_data = LanguagesDataset(_training)

# Transfer data into tensors
q = 3
print(f"generating {q}-grams ", end="")
q_gram_set = set()
language_set = set()

for row in _training:
    text = row["text"]

    for start, end in zip(range(0, len(text)-q), range(q, len(text))):
        q_gram = text[start:end]
        q_gram_set.add(q_gram)

    language_set.add(row["labels"])
    print(".", end="")

q_grams = list(q_gram_set)
languages = list(language_set)
print()
print(f"generating {q}-grams done: {len(q_grams)} generated." )


# Umwandlung languages in Tensoren
print(f"generating {len(languages)} language tensors")

def languageTensor(language):
    startLanguageTensor = timer()
    if language not in language_set:
        raise ValueError(f"Unknown language '{language}' .")
    
    tensor = torch.zeros(1, len(languages))
    i = languages.index(language)
    tensor[0][i] = 1
    
    logging.debug("languageTensor() %f ms" % ((timer()-startLanguageTensor)*1000.0))
    return tensor

language_tensors = dict()
for language in languages:
    language_tensors[language] = languageTensor(language).to(device)




# Umwandlung text in Tensoren
print(f"generating {len(q_gram_set)} qgram tensors")

def qgramTensor(text):
    startQgramTensor = timer()
    tensor = torch.zeros(1, len(q_gram_set))
    for start, end in zip(range(0, len(text)-q), range(q, len(text))):
        _q_gram = text[start:end]
        if _q_gram in q_gram_set:
            i = q_grams.index(_q_gram)
            tensor[0][i] = 1
        else:
            logging.warn(f"Ignoring previously unseen q-gram: '{_q_gram}'")
    
    logging.debug("qgramTensor() %f ms" % ((timer()-startQgramTensor)*1000.0))
    return tensor

qgram_tensors = dict()
samples = list(range(len(training_data)))
for i in tqdm(samples, unit="sample", desc="Generate Tensor"):
    label, text = training_data[i]
    qgram_tensors[text] = qgramTensor(text).to(device)




# Create KNN
input_size = len(q_grams)
output_size = len(languages)
hidden_size = 40

model = FeedForward(input_size, hidden_size, output_size)
print(f"generating NN-model: {model}")
model = model.to(device)
print(f"moved NN-model to device: {device}")

# Definition der Verlustfunktion
criterion = nn.CrossEntropyLoss()

# Back-Propagation
learning_rate = 0.0005
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Implementierung der Trainingsfunktion
def train(category_tensor, text_tensor):
    startTrain = timer()
    model.train()

    output = model(text_tensor[0])

    loss = criterion(output, category_tensor[0])
    loss.backward()

    optimizer.step()

    logging.debug("train() %f ms" % ((timer()-startTrain)*1000.0))
    return output, loss.item()

# Model-Training

EPOCHS = 10
for epoch in tqdm(range(EPOCHS), unit="epoch"):
    samples = list(range(len(training_data)))
    random.shuffle(samples)
    for i in tqdm(samples, unit="sample", desc="Training"):
        label, text = training_data[i]

        language = language_tensors[label] # languageTensor(label)
        #language = language.to(device)

        input = qgram_tensors[text] # qgramTensor(text)
        #input = input.to(device)

        output, loss = train(language, input)
