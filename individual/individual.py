import torch
from sklearn.metrics import root_mean_squared_error


class Individual():

    def __init__(self, structure,  total_train_semantics = None, total_test_semantics = None,
                 hidden_inputs = None, hidden_test_inputs = None):

        self.structure = structure
        self.size = sum([block.size for block in self.structure])
        self.length = len(structure)
        
        self.total_train_semantics = total_train_semantics
        self.total_test_semantics = total_test_semantics

        self.train_semantics = None if self.total_train_semantics is None else torch.sum(self.total_train_semantics, dim = 1)
        self.test_semantics = None if self.total_test_semantics is None else torch.sum(self.total_test_semantics, dim = 1)

        self.hidden_inputs = hidden_inputs
        self.hidden_test_inputs = hidden_test_inputs

        
    def calculate_semantics(self, X, test):
        
        if test and self.total_train_semantics is None:
            self.total_test_semantics = [block.calculate_semantics(X, test) for block in self.structure]
            self.test_semantics = torch.sum(self.test_semantics, dim = 1)
        
        elif not test and self.total_train_semantics is None:
            self.total_train_semantics = [block.calculate_semantics(X, test) for block in self.structure]
            self.train_semantics = torch.sum(self.train_semantics, dim = 1)

    def evaluate(self, y, metric = root_mean_squared_error, test = False):

        if not test:

            self.fitness = metric(y, self.train_semantics)

        else:

            self.test_fitness = metric(y, self.test_semantics)

    def create_hidden_input(self, X, test=False):

        if test:

            if self.structure[0].hidden_test_inputs is None:
                self.structure[0].create_hidden_input(X, test)

            self.hidden_test_inputs = self.structure[0].hidden_test_inputs

        else:

            if self.structure[0].hidden_inputs is None:
                self.structure[0].create_hidden_input(X, test)

            self.hidden_inputs = self.structure[0].hidden_inputs