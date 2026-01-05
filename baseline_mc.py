from NeuralNetwork.utils import create_network
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from datasets.data_loader_resnet import *
from utils.utils import StandardScaler
from utils.evaluators import cross_entropy
from utils.info import logger
from torch import nn
import torch
import torch.optim as optim
import datetime
import time
from sklearn.model_selection import train_test_split

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

loaders = [
    load_mc_tuberculosis
]

resnet_versions = ['rn18', 'rn34', 'rn50']
sample_sizes = [10, 50, 100, 250, 500]

# Baseline architecture configurations for multiclass
# Using reasonable defaults - can be tuned based on evolutionary results
config = {'mc_tuberculosis_rn18_100_cross_entropy': {'width': 4, 'depth': 641},
 'mc_tuberculosis_rn18_10_cross_entropy': {'width': 4, 'depth': 601},
 'mc_tuberculosis_rn18_250_cross_entropy': {'width': 4, 'depth': 589},
 'mc_tuberculosis_rn18_500_cross_entropy': {'width': 4, 'depth': 640},
 'mc_tuberculosis_rn18_50_cross_entropy': {'width': 4, 'depth': 594},
 'mc_tuberculosis_rn34_100_cross_entropy': {'width': 4, 'depth': 632},
 'mc_tuberculosis_rn34_10_cross_entropy': {'width': 4, 'depth': 605},
 'mc_tuberculosis_rn34_250_cross_entropy': {'width': 4, 'depth': 591},
 'mc_tuberculosis_rn34_500_cross_entropy': {'width': 4, 'depth': 632},
 'mc_tuberculosis_rn34_50_cross_entropy': {'width': 4, 'depth': 593},
 'mc_tuberculosis_rn50_100_cross_entropy': {'width': 4, 'depth': 541},
 'mc_tuberculosis_rn50_10_cross_entropy': {'width': 4, 'depth': 512},
 'mc_tuberculosis_rn50_250_cross_entropy': {'width': 4, 'depth': 576},
 'mc_tuberculosis_rn50_500_cross_entropy': {'width': 4, 'depth': 598},
 'mc_tuberculosis_rn50_50_cross_entropy': {'width': 4, 'depth': 573}}

seeds = 5


def _run(seed, loader, resnet_v, sample_size):
    """
    Train a baseline neural network for multiclass classification.

    Args:
        seed: Random seed for reproducibility
        loader: Data loader function
        resnet_v: ResNet version ('rn18', 'rn34', 'rn50')
        sample_size: Number of samples in test set
    """
    X, y = loader(model_type=resnet_v, X_y=True)
    dataset_base = loader.__name__.split("load_")[-1]
    dataset = f"{dataset_base}_{resnet_v}_{sample_size}_cross_entropy"

    # Get number of classes
    n_classes = int(y.unique().shape[0])

    # Standardize features
    X = StandardScaler().fit_transform(X)

    # Use baseline configuration
    width = config['dataset']['width']
    depth = config['dataset']['depth']

    # Create network for multiclass classification (no output activation, use CrossEntropyLoss)
    net = NeuralNetwork(create_network(X.shape[1],
                                       width=1,
                                       depth=depth*width,
                                       num_outputs=n_classes,
                                       output_activation=None))  # No activation - CrossEntropyLoss includes softmax

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sample_size, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

    # Convert to tensors
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).long()  # CrossEntropyLoss expects long
    y_test = torch.Tensor(y_test).long()
    # X_train_nn = torch.Tensor(X_train_nn)
    X_val = torch.Tensor(X_val)
    # y_train_nn = torch.Tensor(y_train_nn).long()
    y_val = torch.Tensor(y_val).long()

    start = time.time()

    # Train with CrossEntropyLoss for multiclass classification
    net.train_network(X_train, y_train,
                      X_val=X_val, y_val=y_val,
                      X_test=X_test, y_test=y_test,
                      epochs=2000, batch_size=32,
                      criterion=nn.CrossEntropyLoss(),
                      optimizer=optim.Adam)

    end = time.time()

    # Log evolution of training
    for epoch, (train_loss, val_loss, test_loss) in enumerate(zip(net.history["train_loss"],
                                                                    net.history["val_loss"],
                                                                    net.history["test_loss"])):
        logger(f'log/baseline_mc_evo_{day}.csv',
               generation=epoch,
               timing=0,
               run_info=[dataset, train_loss, val_loss, test_loss],
               seed=seed)

    # Compute final cross-entropy on test set
    y_pred = net.forward(X_test)
    test_ce = cross_entropy(y_test, y_pred)

    # Log final results
    logger(f'log/baseline_mc_{day}.csv',
           generation=1000,
           timing=end - start,
           run_info=[dataset, test_ce],
           seed=seed)

    return 'done'