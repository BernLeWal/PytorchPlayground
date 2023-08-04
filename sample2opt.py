#!/bin/python

import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
import random
from tqdm.autonotebook import tqdm


# GLOBAL
_device = "cuda:0" if (torch.has_cuda) else ("mps" if torch.has_mps else "cpu")
logging.basicConfig(filename="sample2opt.log", level=logging.DEBUG)

_q = 3 # q-gram size
_criterion = nn.CrossEntropyLoss()  # loss-function to be used
_learning_rate = 0.0005



## Class-Definition Feed-Forward net with one hidden layer
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


## Implementation of the Data-Class
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


# Load data and create caches for labels and entities
def loadData(filename):
    # Load JSON file
    print("loading %s ..." % (filename))
    with open(filename, "rt", encoding="utf8") as train_file:
        training = json.load(train_file)
    print(f"loading done: loaded {len(training)} rows.")
    return training

def collectLabelsAndEntities(training):
    # Collect existing labels and datas
    print("collection lables and entities ...")
    label_set = set()
    entity_set = set()
    for row in training:
        text = row["text"]
        for start, end in zip(range(0, len(text)-_q), range(_q, len(text))):
            entity_set.add(text[start:end])
        label_set.add(row["labels"])
    print(f"collection done: {len(label_set)} labels, {len(entity_set)} entities" )
    
    # Build index caches
    label2index = dict()
    label_index = 0
    for label in label_set:
        label2index[label] = label_index
        label_index = label_index + 1
    print(f"built index-cache for {len(label2index)} labels")

    entity2index = dict()
    entity_index = 0
    for entity in entity_set:
        entity2index[entity] = entity_index
        entity_index = entity_index + 1
    print(f"built index-cache for {len(entity2index)} entities")

    return label2index, entity2index


# Build Label-Tensors
def buildLabelTensor(label, label2index):
    startBuildLabelTensor = timer()
    if label not in label2index:
        raise ValueError(f"Unknown language '{label}' .")
    
    labelTensor = torch.zeros(1, len(label2index))
    labelTensor[0][label2index[label]] = 1

    logging.debug("buildLabelTensor() %f ms" % ((timer()-startBuildLabelTensor)*1000.0))
    return labelTensor

def buildLabelTensors(labels):
    label_tensors = dict()
    for label in labels:
        label_tensors[label] = buildLabelTensor(label, labels).to(_device)
    return label_tensors


# Build Entity-Tensors
def buildEntityTensor(text, entity2index):
    startBuildEntityTensor = timer()
    entityTensor = torch.zeros(1, len(entity2index))
    for start, end in zip(range(0, len(text)-_q), range(_q, len(text))):
        entity = text[start:end]
        if entity in entity2index:
            entityTensor[0][entity2index[entity]] = 1
        else:
            logging.warn(f"Ignoring previously unseen entity: '{entity}'")

    logging.debug("buildEntityTensor() %f ms" % ((timer()-startBuildEntityTensor)*1000.0))
    return entityTensor

def buildEntityTensors(training_data, entities):
    entity_tensors = dict()
    samples = list(range(len(training_data)))
    for i in tqdm(samples, unit="sample", desc="Generate Tensor"):
        _, text = training_data[i]
        entity_tensors[text] = buildEntityTensor(text, entities).to(_device)
    return entity_tensors


# Create KNN
def createKNN(input, output, hidden_size):
    input_size = len(input)
    output_size = len(output)

    model = FeedForward(input_size, hidden_size, output_size)
    print(f"generating NN-model: {model}")
    model = model.to(_device)
    print(f"moved NN-model to device: {_device}")

    return model


# Training
def trainOne(model, optimizer, category_tensor, text_tensor):
    startTrain = timer()
    model.train()

    output = model(text_tensor[0])

    loss = _criterion(output, category_tensor[0])
    loss.backward()

    optimizer.step()

    logging.debug("train() %f ms" % ((timer()-startTrain)*1000.0))
    return output, loss.item()

def trainAll(epochs, training_data, labelTensors, entityTensors, model, optimizer):
    for epoch in tqdm(range(epochs), unit="epoch"):
        samples = list(range(len(training_data)))
        random.shuffle(samples)
        for i in tqdm(samples, unit="sample", desc="Training"):
            label, text = training_data[i]

            label = labelTensors[label] 
            input = entityTensors[text] 

            output, loss = trainOne(model, optimizer, label, input)


# MAIN
def main():
    print("Language Classification with PyTorch (speed optimized):")
    print(f"Will use device: {_device}")

    training = loadData("data/test.json")
    training_data = LanguagesDataset(training)
    labels, entities = collectLabelsAndEntities(training)
    labelTensors = buildLabelTensors(labels)
    entityTensors = buildEntityTensors(training_data, entities)

    model = createKNN(entities, labels, 40)
    optimizer = optim.SGD(model.parameters(), lr=_learning_rate)
    trainAll(10, training_data, labelTensors, entityTensors, model, optimizer)



if __name__ == "__main__":
    main()
