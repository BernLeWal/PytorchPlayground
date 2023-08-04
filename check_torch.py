#!/bin/python
import torch

if torch.has_cuda:
    device = "cuda:0"
elif torch.has_mps:
    device = "mps"
else:
    device = "cpu"

print(device)