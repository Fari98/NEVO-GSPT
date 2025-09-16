import random

from NeuralNetwork.utils import create_random_network, _train_network
from torch import nn
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from joblib import Parallel, delayed
import torch.optim as optim


def initialize_population(X_train, y_train,
                          maximum_width = 6,
                          maximum_depth = 2,
                          activation_functions = [nn.ReLU()],
                          pretrain_part = 1,
                          X_val=None, y_val=None,
                          epochs=100, batch_size=32, learning_rate=0.001,
                          criterion=nn.MSELoss(), optimizer=optim.Adam, device='cpu'

                          ):

    def initializer(pop_size,  n_jobs = 1):

        population = [NeuralNetwork(*create_random_network(X_train.shape[1], maximum_width,
                                              maximum_depth, activation_functions)) for _ in range(pop_size)]

        num_to_train = int(pretrain_part * len(population))

        # Select individuals to train
        individuals_to_train = population[:num_to_train]


        trained_individuals = Parallel(n_jobs=n_jobs)(
            delayed(_train_network)(
                individual,
                X_train, y_train,
                X_val, y_val,
                None, None,
                epochs, batch_size, learning_rate,
                criterion, optimizer, device
            ) for individual in individuals_to_train
        )

        population[:num_to_train] = trained_individuals

        [ind.__setattr__('trained', True) for ind in population]

        return population

    return initializer

