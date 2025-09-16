import torch
import numpy as np
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.utils import parameters_to_vector as p2v


class Block():


    def __init__(self, layers, hidden_train_inputs, ms, hidden_test_inputs = None):

        self.layers = layers
        self.model = nn.Sequential(layers)

        self.size = p2v(self.model.parameters()).numel()

        self.hidden_train_inputs = hidden_train_inputs
        self.hidden_test_inputs = hidden_test_inputs
        self.ms = ms

        self.train_semantics = None
        self.test_semantics = None

    def forward(self, X, test = False, new_hidden_data = None):

        layers = list(filter(lambda x: 'hidden' in x, self.layers.keys()))
        activations = list(filter(lambda x: 'act' in x, self.layers.keys()))

        if new_hidden_data is None:

            if not test:

                for i, layer in enumerate(layers):
                    X = self.layers[activations[i]](self.layers[layer](X))
                    if not i == (len(layers) - 1):
                        X = torch.cat((X, self.hidden_train_inputs[i]), dim=1)
            else:

                for i, layer in enumerate(layers):
                    X = self.layers[activations[i]](self.layers[layer](X))
                    if not i == (len(layers) - 1):
                        X = torch.cat((X, self.hidden_test_inputs[i]), dim=1)


        else:

            for i, layer in enumerate(layers):
                X = self.layers[activations[i]](self.layers[layer](X))
                if not i == (len(layers)-1):
                    X = torch.cat((X, new_hidden_data[i]), dim = 1)

        return torch.mul(X, self.ms)

    def calculate_semantics(self, X, test = False):

        if test and self.test_semantics is None:
            self.test_semantics = self.forward(X, test)
        elif not test and self.train_semantics is None:
                self.train_semantics = self.forward(X)