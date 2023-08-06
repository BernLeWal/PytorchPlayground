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
logging.basicConfig(level=logging.DEBUG)    # filename="sample2opt.log"

## Settings
_filenameTraining = "data/test.json" # data for training-file in json format
_filenameValidation = "data/valid.json" # data for validation-file in json format

_q = 3 # q-gram size
_hidden_size = 40
_criterion = nn.CrossEntropyLoss()  # loss-function to be used
_learning_rate = 0.0005
_epochs = 10    # number of epochs for training


## Class-Definition Feed-Forward net with one hidden layer
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.hidden = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y_1 = self.input_layer(x)
        y_non_linear = self.sigmoid(y_1)
        out = self.hidden(y_non_linear)
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
    if label not in label2index:
        raise ValueError(f"Unknown language '{label}' .")
    
    labelTensor = torch.zeros(1, len(label2index))
    labelTensor[0][label2index[label]] = 1
    return labelTensor

def buildLabelTensors(labels):
    label_tensors = dict()
    for label in labels:
        label_tensors[label] = buildLabelTensor(label, labels).to(_device)
    return label_tensors


# Build Entity-Tensors
def buildEntityTensor(text, entity2index):
    entityTensor = torch.zeros(1, len(entity2index))
    for start, end in zip(range(0, len(text)-_q), range(_q, len(text))):
        entity = text[start:end]
        if entity in entity2index:
            entityTensor[0][entity2index[entity]] = 1
#        else:
#            logging.warn(f"Ignoring previously unseen entity: '{entity}'")
    return entityTensor

def buildEntityTensors(training_data, entities):
    entity_tensors = dict()
    samples = list(range(len(training_data)))
    for i in tqdm(samples, unit="sample", desc="Generate Tensors"):
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
    model.train()

    output = model(text_tensor[0])

    loss = _criterion(output, category_tensor[0])
    loss.backward()

    optimizer.step()

    return output, loss.item()

def trainAll(epochs, training_data, labelTensors, entityTensors, model, optimizer, validation_data, labels, validationTensors):
    for epoch in tqdm(range(epochs), unit="epoch"):
        samples = list(range(len(training_data)))
        random.shuffle(samples)
        for i in tqdm(samples, unit="sample", desc="Training"):
            label, text = training_data[i]

            label = labelTensors[label] 
            input = entityTensors[text] 

            output, loss = trainOne(model, optimizer, label, input)
            #print(f"Loss:\t{str(loss)}")

        total, correct = evaluateAll(validation_data, model, labels, validationTensors)
        print(f"Samples evaluated:\t{total}")
        print(f"Samples correct:\t{correct}")
        print(f"Accuracy:\t{correct/total}")


# Evaluation
def evaluateOne(model, validation_tensor):
    output = model(validation_tensor[0])
    return output

def evaluateAll(dataset, model, labels, validationTensors):
    index2label = dict()
    for key in labels:
        index2label[labels[key]] = key

    model.eval()

    total = 0.0
    correct = 0.0
    for i in tqdm(range(len(dataset)), unit="sample", desc="Evaluation"):
        label, text = dataset[i]
        input = validationTensors[text]

        output = evaluateOne(model, input)
        predicted_item = torch.argmax(output).item()
        predicted = index2label[predicted_item]

        total += 1.0
        correct += predicted == label
    return total, correct


# MAIN
def main():
    start = timer()
    print("Language Classification with PyTorch (speed optimized):")
    print(f"Will use device: {_device}")

    training = loadData(_filenameTraining)
    training_data = LanguagesDataset(training)
    validation = loadData(_filenameValidation)
    validation_data = LanguagesDataset(validation)

    labels, entities = collectLabelsAndEntities(training)
    labelTensors = buildLabelTensors(labels)
    entityTensors = buildEntityTensors(training_data, entities)
    validationTensors = buildEntityTensors(validation_data, entities)

    model = createKNN(entities, labels, _hidden_size)
    optimizer = optim.SGD(model.parameters(), lr=_learning_rate)
    trainAll(_epochs, training_data, labelTensors, entityTensors, model, optimizer, validation_data, labels, validationTensors)
    print("Total time %f s" % ((timer()-start)))


if __name__ == "__main__":
    main()
