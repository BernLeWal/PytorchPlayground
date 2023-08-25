#!/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from timeit import default_timer as timer
from DatasetLoader import DatasetLoader
from LanguagesDataset import LanguagesDataset
from FeedForward import FeedForward
from tqdm.autonotebook import tqdm

"""
Sample-Code originaly based on the iX-article "Einstieg in PyTorch" (see https://ix.de/zbu8)
Separated the Workflow implementation into distinct classes.
"""

# GLOBAL
logging.basicConfig(level=logging.DEBUG)    # filename="logs/sample2opt.log"


class LanguagesNN:

    device = "cuda:0" if (torch.has_cuda) else ("mps" if torch.has_mps else "cpu")

    def __init__(self, epochs : int = 10, hiddenSize : int = 40, learningRate : float = 0.0005):
        self._hiddenSize = hiddenSize
        self._learningRate = learningRate
        self._epochs = epochs

        self._criterion = nn.CrossEntropyLoss()  # loss-function to be used

        self._trainingLoader : DatasetLoader
        self._trainingDataset : LanguagesDataset
        self._validationLoader : DatasetLoader
        self._validationDateset : LanguagesDataset
        self._model : FeedForward


    def loadTrainingData(self, fileName : str, limit = -1, preShuffle = False, postShuffle = False):
        self._trainingLoader = DatasetLoader( fileName, limit, preShuffle, postShuffle )
        data = self._trainingLoader.loadData()
        self._trainingDataset = LanguagesDataset(data)
        return self._trainingDataset


    def loadValidationData(self, fileName : str, limit = -1, preShuffle = False, postShuffle = False):
        self._validationLoader = DatasetLoader( fileName, limit, preShuffle, postShuffle )
        data = self._validationLoader.loadData()
        self._validationDataset = LanguagesDataset(data)
        return self._validationDataset


    def _training(self, labelTensors, entityTensors):
        self._model.train()
        samples = list(range(len(self._trainingDataset)))
        random.shuffle(samples)
        for i in tqdm(samples, unit="sample", desc="Training"):
            label, text = self._trainingDataset[i]
            label = labelTensors[label] 
            input = entityTensors[text] 

            output = self._model(input[0])
            loss = self._criterion(output, label[0])
            loss.backward()
            self._optimizer.step()

            #print(f"Loss:\t{str(loss.item)}")


    def _evaluate(self, validationTensors, index2label):
        self._model.eval()
        total = 0.0
        correct = 0.0
        for i in tqdm(range(len(self._validationDataset)), unit="sample", desc="Evaluation"):
            label, text = self._validationDataset[i]
            input = validationTensors[text]

            output = self._model(input[0])
            predicted_item = torch.argmax(output).item()
            predicted = index2label[predicted_item]

            total += 1.0
            correct += predicted == label
        return total, correct


    def run(self):
        start = timer()

        # Build up caches (to speed up operation)
        labels, entities = self._trainingLoader.collectLabelsAndEntities()
        index2label = dict()
        for key in labels:
            index2label[labels[key]] = key

        # Build Tensors
        labelTensors = self._trainingLoader.buildLabelTensors()
        entityTensors = self._trainingLoader.buildEntityTensors()
        validationTensors = self._validationLoader.buildEntityTensors(entities)

        # Create the model (the NN)
        self._model = FeedForward(len(entities), self._hiddenSize, len(index2label))
        print(f"generating NN-model: {self._model}")
        self._model = self._model.to(LanguagesNN.device)
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._learningRate)

        # Perform training and evaluation (after every epoch)
        for epoch in tqdm(range(self._epochs), unit="epoch"):
            self._training(labelTensors, entityTensors)
            total, correct = self._evaluate(validationTensors, index2label)
            print(f"Samples evaluated: \t{correct} of {total} correct. (Accuracy:{correct/total})")

        print("Total time %f s" % ((timer()-start)))


# MAIN
def main():
    print("Language Classification with PyTorch (speed optimized):")
    print(f"Will use device: {LanguagesNN.device}")


    languagesNN = LanguagesNN(10, 40, 0.0005)
    languagesNN.loadTrainingData("data/train.json", 10000)
    languagesNN.loadValidationData("data/test.json", 1000)
    languagesNN.run()


if __name__ == "__main__":
    main()