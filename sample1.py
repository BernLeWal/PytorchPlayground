#!/bin/python
import torch
import torch.nn as nn

# Erzeugt ein einfaches Feed-Forward-Netz, 
# das einen Input ohne Weiteres durch einen oder mehrere lineare Layer eines KNN schleust.
# - die neue Klasse erbt von torch.nn.Module
# - ctor definiert die Komponenten des Netzes;
#   hier ein einzelner linearer Layer (self.linear)
# - die forward()-Methode nimmt einen kontreten Input an, 
#   schleust ihn durchs Netz (Forward Pass) und
#   gibt den entsprechenden Output zurück

class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    

model = FeedForward(256,8)

print(model)
