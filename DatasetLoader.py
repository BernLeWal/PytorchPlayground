#!/bin/python
import json
import torch
from typing import List
import random
from tqdm.autonotebook import tqdm

class DatasetLoader:
    def __init__(self, filename, limit=-1, preShuffle=False, postShuffle=False) -> None:
        self._filename = filename
        self._limit = limit
        self._preShuffle = preShuffle
        self._postShuffle = postShuffle

        self._q = 3 # q-gram size

        self._device = "cuda:0" if (torch.has_cuda) else ("mps" if torch.has_mps else "cpu")
        self._data = None
        self._label2index = dict()
        self._entity2index = dict()


    def loadData(self) -> List[any]:
        if(self._filename.endswith(".csv")):
            data = self._loadDataCSV()
        else:
            data = self._loadDataJSON()

        if(self._preShuffle):
            random.shuffle(data)

        if(self._limit>0 and len(data)>self._limit):
            data = data[:self._limit]
            print(f"truncated list to: {len(data)} elements.")

        if(self._postShuffle):
            random.shuffle(data)

        self._data = data
        return data


    def _loadDataJSON(self) -> List[any]:
        # Load JSON file
        print("loading %s ..." % (self._filename))
        with open(self._filename, "rt", encoding="utf8") as train_file:
            training = json.load(train_file)
        print(f"loading done: loaded {len(training)} rows.")
        return training
    

    def _loadDataCSV(self) -> List[any]:
        return None
    

    def collectLabelsAndEntities(self):
        # Collect existing labels and datas
        print("collection lables and entities ...")
        label_set = set()
        entity_set = set()
        for row in self._data:
            text = row["text"]
            for start, end in zip(range(0, len(text)-self._q), range(self._q, len(text))):
                entity_set.add(text[start:end])
            label_set.add(row["labels"])
        print(f"collection done: {len(label_set)} labels, {len(entity_set)} entities" )
        
        # Build index caches
        label_index = 0
        for label in label_set:
            self._label2index[label] = label_index
            label_index = label_index + 1
        print(f"built index-cache for {len(self._label2index)} labels")

        entity_index = 0
        for entity in entity_set:
            self._entity2index[entity] = entity_index
            entity_index = entity_index + 1
        print(f"built index-cache for {len(self._entity2index)} entities")

        return self._label2index, self._entity2index
    


    # Build Label-Tensors
    def _buildLabelTensor(self, label):
        if label not in self._label2index:
            raise ValueError(f"Unknown language '{label}' .")
        
        labelTensor = torch.zeros(1, len(self._label2index))
        labelTensor[0][self._label2index[label]] = 1
        return labelTensor

    def buildLabelTensors(self):
        label_tensors = dict()
        for label in self._label2index:
            label_tensors[label] = self._buildLabelTensor(label).to(self._device)
        return label_tensors


    # Build Entity-Tensors
    def _buildEntityTensor(self, text, entity2index):
        entityTensor = torch.zeros(1, len(entity2index))
        for start, end in zip(range(0, len(text)-self._q), range(self._q, len(text))):
            entity = text[start:end]
            if entity in entity2index:
                entityTensor[0][entity2index[entity]] = 1
    #        else:
    #            logging.warn(f"Ignoring previously unseen entity: '{entity}'")
        return entityTensor

    def buildEntityTensors(self, entity2index = None):
        if entity2index is None:
            entity2index = self._entity2index

        entity_tensors = dict()
        samples = list(range(len(self._data)))
        for i in tqdm(samples, unit="sample", desc="Generate Tensors"):
            text = self._data[i]["text"]
            entity_tensors[text] = self._buildEntityTensor(text, entity2index).to(self._device)
        return entity_tensors
