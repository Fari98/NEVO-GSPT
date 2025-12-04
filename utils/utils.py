import random
import math
from sklearn.metrics import matthews_corrcoef
import numpy as np

def uniform_random_step_generator(lower, upper):

    def ursg():

        return random.uniform(lower, upper)

    return ursg

import torch


class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0, stratify=None):
    """ Splits X and y tensors into train and test subsets

    This method replicates the behaviour of Sklearn's 'train_test_split' with optional stratification.

    Parameters
    ----------
    X : torch.Tensor
        Input data instances,
    y : torch.Tensor
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether or not to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether or not to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random numbers generators.
    stratify : torch.Tensor or None (default=None)
        If a tensor is provided, split the data in a stratified fashion, using this as the class labels.
        If None, performs regular split without stratification.

    Returns
    -------
    X_train : torch.Tensor
        Training data instances.
    y_train : torch.Tensor
        Training target vector.
    X_test : torch.Tensor
        Test data instances.
    y_test : torch.Tensor
        Test target vector.
    train_indices : torch.Tensor
        Indices representing the training partition.
    test_indices : torch.Tensor
    Indices representing the test partition.
    """
    # Sets the seed before generating partition's indexes
    torch.manual_seed(seed)
    np.random.seed(seed)

    if stratify is not None:
        # Use stratified split to maintain class distribution (pure torch implementation)
        stratify_tensor = stratify.flatten()

        # Get unique classes
        unique_classes = torch.unique(stratify_tensor)
        train_indices_list = []
        test_indices_list = []

        # For each class, split stratified
        for class_label in unique_classes:
            class_mask = stratify_tensor == class_label
            class_indices = torch.where(class_mask)[0]

            # Shuffle indices for this class
            torch.manual_seed(seed)
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]

            # Split this class
            split_point = int(torch.ceil(torch.tensor(len(shuffled_indices) * p_test)))
            test_idx = shuffled_indices[:split_point]
            train_idx = shuffled_indices[split_point:]

            train_indices_list.append(train_idx)
            test_indices_list.append(test_idx)

        # Concatenate all indices
        train_indices = torch.cat(train_indices_list)
        test_indices = torch.cat(test_indices_list)

        # Shuffle if requested
        if shuffle:
            torch.manual_seed(seed)
            train_perm = torch.randperm(len(train_indices))
            test_perm = torch.randperm(len(test_indices))
            train_indices = train_indices[train_perm]
            test_indices = test_indices[test_perm]
    else:
        # Original non-stratified split
        if shuffle:
            indices = torch.randperm(X.shape[0])
        else:
            indices = torch.arange(0, X.shape[0], 1)
        # Splits indices
        split = int(math.floor(p_test * X.shape[0]))
        train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        # Generates train/test partitions
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test

