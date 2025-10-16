from NeuralNetwork.utils import create_network
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from datasets.data_loader import *
from utils.utils import StandardScaler, train_test_split, rmse, mse
from utils.info import logger
import datetime
import time

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

loaders = [
    load_bioav,
    load_ld50,
    load_concrete_strength,
    load_airfoil
]
size = [
    (4,299),
    (4,450),
    (3,244),
    (3,131)
]
seeds = 5

# for i, loader in enumerate(loaders):
#
# for seed in range(10):


def _run(seed, loader, idx):

    X, y = loader(X_y=True)
    dataset = loader.__name__.split("load_")[-1]
    X = StandardScaler().fit_transform(X)

    net = NeuralNetwork(create_network(X.shape[1], width=size[idx][0], depth=size[idx][1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, p_test=0.25, seed=seed)

    start = time.time()


    net.train_network(X_train, y_train,
                      X_val = X_val, y_val=y_val,
                      X_test=X_test, y_test=y_test,
                      epochs = 1000, batch_size=32 if dataset != 'ld50' else 4) # [4, 100, X_train.shape[0]]

    end = time.time()

    for epoch, (train_loss, val_loss, test_loss) in enumerate(zip(net.history["train_loss"], net.history["val_loss"], net.history["test_loss"])):

        logger(f'log/baseline_dropout_evo_{day}.csv',
               generation=epoch,
               timing=0,
               run_info=[dataset , train_loss, val_loss, test_loss],
               seed=seed)

    y_pred = net.forward(X_test).flatten()

    perf = rmse(y_test, y_pred)

    logger(f'log/baseline_dropout_{day}.csv',
           generation=1000,
           timing = end - start,
           run_info = [dataset , perf],
           seed = seed)




