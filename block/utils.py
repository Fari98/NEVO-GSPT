import torch
from torch import nn
import numpy as np
import random
from collections import OrderedDict


class Linear(torch.nn.Module):
    # a linear activation function based on y=x
    def forward(self, output):
        return output
def create_random_block(input_shape,
                        layers_dim,
                        activation_functions = [nn.ReLU()],
                        neuron_probability = 1):


    layer = nn.Linear(input_shape, 1, bias = False)
    layer.weight.data.uniform_(-1.0, 1.0)
    mask = [i for i in range(input_shape) if random.randint(0, 1) == 0]
    layer.weight.data[:, mask] = 0.0
    # layer.bias.data.fill_(0)

    layers = [('hidden1', layer),
              ('act1', random.choice(activation_functions))]

    for i, dim in enumerate(layers_dim):

        #todo randomize this process if random.random() < p do it else pass(if p == 1 normal)

        p = random.random()

        layer = nn.Linear(1+dim, 1, bias=False)
        if p < neuron_probability:
            layer.weight.data.uniform_(-1.0, 1.0)
        else:
            layer.weight.data.fill_(0) #todo adjust to [0, 0, ... 1]
            layer.weight.data[0][-1] = 1
        # layer.bias.data.fill_(0)

        if i == (len(layers_dim)-1): #last layer
            layers.extend([(f'hidden{i + 2}', layer),
                           (f'act{i + 2}', nn.Tanh())])
        else:
            if p < neuron_probability:
                layers.extend([(f'hidden{i+2}', layer),
                        (f'act{i+2}', random.choice(activation_functions))])
            else:
                layers.extend([(f'hidden{i+2}', layer),
                           (f'act{i + 2}', Linear())])  #todo linear activation function

    # layers.extend([(f'hidden{i+3}', nn.Linear(1+dim, 1)),
    #                 (f'act{i+3}', nn.Tanh())])

    return OrderedDict(layers)