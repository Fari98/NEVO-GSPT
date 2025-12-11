import torch
from sklearn.metrics import root_mean_squared_error


class Individual():

    def __init__(self, structure,  total_train_semantics = None, total_test_semantics = None,
                 hidden_inputs = None, hidden_test_inputs = None, multiclass = False, n_classes = 1, layers_dim = None):

        self.structure = structure
        self.size = sum([block.size for block in self.structure])
        self.length = len(structure)
        self.multiclass = multiclass
        self.n_classes = n_classes

        # Store layers_dim from the first neural network in the structure
        if layers_dim is None:
            # Try to get layers_dim from the first element if it has it
            if hasattr(structure[0], 'layers_dim'):
                self.layers_dim = structure[0].layers_dim
            else:
                self.layers_dim = None
        else:
            self.layers_dim = layers_dim

        self.total_train_semantics = total_train_semantics
        self.total_test_semantics = total_test_semantics

        self.train_semantics = None if self.total_train_semantics is None else self._compute_semantics(self.total_train_semantics, multiclass, n_classes)

        self.test_semantics = None if self.total_test_semantics is None else self._compute_semantics(self.total_test_semantics, multiclass, n_classes)

        self.hidden_inputs = hidden_inputs
        self.hidden_test_inputs = hidden_test_inputs

    def _compute_semantics(self, total_semantics, multiclass, n_classes):
        """
        Compute final semantics from total semantics of all blocks.

        For regression/binary: sum all block outputs along dim=1
        For multiclass: reshape [batch, n_blocks*n_classes] to [batch, n_blocks, n_classes], then sum along dim=1
        """
        if not multiclass:
            # Regression/binary classification: sum all outputs
            return torch.sum(total_semantics, dim=1)
        else:
            # Multiclass: need to reshape and sum properly
            if len(total_semantics.shape) >= 3:
                # Already 3D [batch, n_blocks, n_classes]
                return torch.sum(total_semantics, dim=1)
            elif len(total_semantics.shape) == 2:
                # 2D [batch, n_blocks*n_classes] - need to reshape
                batch_size = total_semantics.shape[0]
                total_features = total_semantics.shape[1]

                # Calculate number of blocks
                n_blocks = total_features // n_classes

                if n_blocks * n_classes == total_features:
                    # Reshape to [batch, n_blocks, n_classes] and sum
                    reshaped = total_semantics.view(batch_size, n_blocks, n_classes)
                    return torch.sum(reshaped, dim=1)
                else:
                    # Dimension mismatch - just return as-is (shouldn't happen)
                    return total_semantics
            else:
                return total_semantics


    def calculate_semantics(self, X, test, multiclass = False):
        
        if test and self.total_test_semantics is None:
            self.total_test_semantics = [block.calculate_semantics(X, test) for block in self.structure]
            self.test_semantics = torch.sum(self.total_test_semantics, dim = 1) if not multiclass else \
            (torch.sum(self.total_test_semantics, dim = 2) if len(self.total_test_semantics.shape) >= 3 else self.total_test_semantics)
        
        elif not test and self.total_train_semantics is None:
            self.total_train_semantics = [block.calculate_semantics(X, test) for block in self.structure]
            self.train_semantics = torch.sum(self.total_train_semantics, dim=1) if not multiclass else \
                (torch.sum(self.total_train_semantics, dim=2) if len(
                    self.total_train_semantics.shape) >= 3 else self.total_train_semantics)

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