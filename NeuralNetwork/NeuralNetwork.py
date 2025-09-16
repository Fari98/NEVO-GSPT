from torch import nn
import torch.optim as optim
from NeuralNetwork.utils import _train_network
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.utils import parameters_to_vector as p2v

class NeuralNetwork(nn.Module):

    def __init__(self, layers, layers_dim = None):
        super().__init__()
        self.layers = layers
        self.model = nn.Sequential(layers)

        self.size = p2v(self.model.parameters()).numel()

        self.layers_dim = layers_dim if layers_dim is not None else None #todo implement automatic dim calculator ?

        self.trained = False #todo add possibility to load weights ?

        self.hidden_test_inputs = None
        self.hidden_inputs = None

    def forward(self, X):
        return self.model(X)

    def train_network(self,
              X_train, y_train,
              X_val=None, y_val=None,
              X_test=None, y_test=None,
              epochs=100, batch_size=32, learning_rate=0.001,
              criterion=nn.MSELoss(), optimizer=optim.Adam, device='cpu'):

        self.history = _train_network(self, X_train, y_train,
              X_val, y_val,
              X_test, y_test,
              epochs, batch_size, learning_rate,
              criterion, optimizer, device, return_history=True)



    def get_hidden_state(self, layer, input):

        if self.trained:

            inner_model = create_feature_extractor(self.model, return_nodes={layer: "hidden_output"})
            return inner_model(input)["hidden_output"]
        else:
            raise Exception('Model should be trained')

    def create_hidden_input(self, X, test = False):

        hidden_inputs = []
        for layer in list(filter(lambda x: 'hidden' in x, self.layers.keys())):
            hidden_inputs.append(self.get_hidden_state(layer, X))

        if test:

            self.hidden_test_inputs = hidden_inputs
        else:

            self.hidden_inputs = hidden_inputs