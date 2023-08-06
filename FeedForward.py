#!/bin/python
import torch.nn as nn

## Class-Definition Feed-Forward net with one hidden layer

# Erzeugt ein einfaches Feed-Forward-Netz, 
# das einen Input ohne Weiteres durch einen oder mehrere lineare Layer eines KNN schleust.
# - die neue Klasse erbt von torch.nn.Module
# - ctor definiert die Komponenten des Netzes;
#   hier ein einzelner linearer Layer (self.linear)
# - die forward()-Methode nimmt einen kontreten Input an, 
#   schleust ihn durchs Netz (Forward Pass) und
#   gibt den entsprechenden Output zurück
class FeedForward(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, output_size : int):
        super(FeedForward, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.hidden = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y_1 = self.input_layer(x)
        y_non_linear = self.sigmoid(y_1)
        out = self.hidden(y_non_linear)
        return out
    

# MAIN
def main():
    model = FeedForward(256, 8, 40)
    print(model)

if __name__ == "__main__":
    main()