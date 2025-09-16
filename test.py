import random

from NeuralNetwork.utils import create_random_network
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from datasets.data_loader import load_concrete_strength
from datasets.utils import train_test_split
from block.block import Block
from block.utils import create_random_block
from individual.individual import Individual
import torch

# Generate some sample regression data
X,y = load_concrete_strength(X_y=True)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


net, layers_dim = create_random_network(X.shape[1])
model = NeuralNetwork(net, layers_dim)

print(net)
print(model)


print(model.trained)
model.train_network(X_train, y_train)
print(model.trained)

# for layer in range
hidden_inputs = []
for layer in list(filter(lambda x: 'hidden' in x, model.layers.keys())):
    hidden_inputs.append(model.get_hidden_state(layer, X_train))

test_hidden_inputs = []
for layer in list(filter(lambda x: 'hidden' in x, model.layers.keys())):
    test_hidden_inputs.append(model.get_hidden_state(layer, X_test))

block = create_random_block(X_train.shape[1], model.layers_dim)

block = Block(block, hidden_inputs, random.uniform(0, 1), test_hidden_inputs)

block.forward(X_train)

print(block)

ind = Individual([model], model.forward(X_train), model.forward(X_test))
print(ind.structure)

ind2 = Individual( ind.structure+[block], torch.cat((ind.total_train_semantics, block.forward(X_train)), dim = 1), torch.cat((ind.total_test_semantics, block.forward(X_test, test=True)), dim = 1))
print(ind2)

ind2 = Individual( ind.structure+[block]+[block], torch.cat((ind.total_train_semantics, block.forward(X_train),block.forward(X_train)), dim = 1), torch.cat((ind.total_test_semantics, block.forward(X_test, test=True), block.forward(X_test, test=True)), dim = 1))
print(ind2)
