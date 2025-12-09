from NeuralNetwork.utils import create_network
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from datasets.data_loader_resnet import *
from utils.utils import StandardScaler, train_test_split
from utils.evaluators import binarized_mcc, binarized_rmse, binarized_bce
from utils.info import logger
from torch import nn
import datetime
import time

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

loaders = [
    load_brain_tumor,
    load_tuberculosis
]

resnet_versions = ['rn18', 'rn34', 'rn50']
sample_sizes = [10, 50, 100, 250, 500]
size = {'depth': {'brain_tumor_rn18_100_bce': 479.0,
  'brain_tumor_rn18_100_brmse': 489.0,
  'brain_tumor_rn18_100_mcc': 20.0,
  'brain_tumor_rn18_10_bce': 477.0,
  'brain_tumor_rn18_10_brmse': 473.0,
  'brain_tumor_rn18_10_mcc': 3.0,
  'brain_tumor_rn18_250_bce': 477.0,
  'brain_tumor_rn18_250_brmse': 475.0,
  'brain_tumor_rn18_250_mcc': 128.0,
  'brain_tumor_rn18_500_bce': 486.0,
  'brain_tumor_rn18_500_brmse': 490.0,
  'brain_tumor_rn18_500_mcc': 174.0,
  'brain_tumor_rn18_50_bce': 476.0,
  'brain_tumor_rn18_50_brmse': 476.0,
  'brain_tumor_rn18_50_mcc': 69.0,
  'brain_tumor_rn34_100_bce': 457.0,
  'brain_tumor_rn34_100_brmse': 465.0,
  'brain_tumor_rn34_100_mcc': 15.0,
  'brain_tumor_rn34_10_bce': 479.0,
  'brain_tumor_rn34_10_brmse': 466.0,
  'brain_tumor_rn34_10_mcc': 2.0,
  'brain_tumor_rn34_250_bce': 481.0,
  'brain_tumor_rn34_250_brmse': 477.0,
  'brain_tumor_rn34_250_mcc': 68.0,
  'brain_tumor_rn34_500_bce': 480.0,
  'brain_tumor_rn34_500_brmse': 481.0,
  'brain_tumor_rn34_500_mcc': 89.0,
  'brain_tumor_rn34_50_bce': 474.0,
  'brain_tumor_rn34_50_brmse': 482.0,
  'brain_tumor_rn34_50_mcc': 15.0,
  'brain_tumor_rn50_100_bce': 474.0,
  'brain_tumor_rn50_100_brmse': 492.0,
  'brain_tumor_rn50_100_mcc': 32.0,
  'brain_tumor_rn50_10_bce': 489.0,
  'brain_tumor_rn50_10_brmse': 489.0,
  'brain_tumor_rn50_10_mcc': 1.0,
  'brain_tumor_rn50_250_bce': 490.0,
  'brain_tumor_rn50_250_brmse': 482.0,
  'brain_tumor_rn50_250_mcc': 92.0,
  'brain_tumor_rn50_500_bce': 477.0,
  'brain_tumor_rn50_500_brmse': 485.0,
  'brain_tumor_rn50_500_mcc': 103.0,
  'brain_tumor_rn50_50_bce': 481.0,
  'brain_tumor_rn50_50_brmse': 478.0,
  'brain_tumor_rn50_50_mcc': 7.0,
  'tuberculosis_rn18_100_bce': 475.0,
  'tuberculosis_rn18_100_brmse': 487.0,
  'tuberculosis_rn18_100_mcc': 43.0,
  'tuberculosis_rn18_10_bce': 472.0,
  'tuberculosis_rn18_10_brmse': 491.0,
  'tuberculosis_rn18_10_mcc': 1.0,
  'tuberculosis_rn18_250_bce': 491.0,
  'tuberculosis_rn18_250_brmse': 493.0,
  'tuberculosis_rn18_250_mcc': 177.0,
  'tuberculosis_rn18_500_bce': 490.0,
  'tuberculosis_rn18_500_brmse': 494.0,
  'tuberculosis_rn18_500_mcc': 271.0,
  'tuberculosis_rn18_50_bce': 465.0,
  'tuberculosis_rn18_50_brmse': 475.0,
  'tuberculosis_rn18_50_mcc': 29.0,
  'tuberculosis_rn34_100_bce': 482.0,
  'tuberculosis_rn34_100_brmse': 472.0,
  'tuberculosis_rn34_100_mcc': 83.0,
  'tuberculosis_rn34_10_bce': 470.0,
  'tuberculosis_rn34_10_brmse': 477.0,
  'tuberculosis_rn34_10_mcc': 1.0,
  'tuberculosis_rn34_250_bce': 494.0,
  'tuberculosis_rn34_250_brmse': 495.0,
  'tuberculosis_rn34_250_mcc': 130.0,
  'tuberculosis_rn34_500_bce': 489.0,
  'tuberculosis_rn34_500_brmse': 483.0,
  'tuberculosis_rn34_500_mcc': 156.0,
  'tuberculosis_rn34_50_bce': 471.0,
  'tuberculosis_rn34_50_brmse': 463.0,
  'tuberculosis_rn34_50_mcc': 14.0,
  'tuberculosis_rn50_100_bce': 483.0,
  'tuberculosis_rn50_100_brmse': 499.0,
  'tuberculosis_rn50_100_mcc': 43.0,
  'tuberculosis_rn50_10_bce': 484.0,
  'tuberculosis_rn50_10_brmse': 473.0,
  'tuberculosis_rn50_10_mcc': 1.0,
  'tuberculosis_rn50_250_bce': 483.0,
  'tuberculosis_rn50_250_brmse': 490.0,
  'tuberculosis_rn50_250_mcc': 115.0,
  'tuberculosis_rn50_500_bce': 478.0,
  'tuberculosis_rn50_500_brmse': 489.0,
  'tuberculosis_rn50_500_mcc': 237.0,
  'tuberculosis_rn50_50_bce': 470.0,
  'tuberculosis_rn50_50_brmse': 479.0,
  'tuberculosis_rn50_50_mcc': 13.0},
 'width': {'brain_tumor_rn18_100_bce': 3.0,
  'brain_tumor_rn18_100_brmse': 3.0,
  'brain_tumor_rn18_100_mcc': 3.0,
  'brain_tumor_rn18_10_bce': 3.0,
  'brain_tumor_rn18_10_brmse': 3.0,
  'brain_tumor_rn18_10_mcc': 4.0,
  'brain_tumor_rn18_250_bce': 4.0,
  'brain_tumor_rn18_250_brmse': 4.0,
  'brain_tumor_rn18_250_mcc': 4.0,
  'brain_tumor_rn18_500_bce': 3.0,
  'brain_tumor_rn18_500_brmse': 3.0,
  'brain_tumor_rn18_500_mcc': 4.0,
  'brain_tumor_rn18_50_bce': 4.0,
  'brain_tumor_rn18_50_brmse': 3.0,
  'brain_tumor_rn18_50_mcc': 3.0,
  'brain_tumor_rn34_100_bce': 3.0,
  'brain_tumor_rn34_100_brmse': 3.0,
  'brain_tumor_rn34_100_mcc': 3.0,
  'brain_tumor_rn34_10_bce': 3.0,
  'brain_tumor_rn34_10_brmse': 3.0,
  'brain_tumor_rn34_10_mcc': 3.0,
  'brain_tumor_rn34_250_bce': 4.0,
  'brain_tumor_rn34_250_brmse': 3.0,
  'brain_tumor_rn34_250_mcc': 4.0,
  'brain_tumor_rn34_500_bce': 4.0,
  'brain_tumor_rn34_500_brmse': 3.0,
  'brain_tumor_rn34_500_mcc': 4.0,
  'brain_tumor_rn34_50_bce': 4.0,
  'brain_tumor_rn34_50_brmse': 3.0,
  'brain_tumor_rn34_50_mcc': 3.0,
  'brain_tumor_rn50_100_bce': 3.0,
  'brain_tumor_rn50_100_brmse': 3.0,
  'brain_tumor_rn50_100_mcc': 3.0,
  'brain_tumor_rn50_10_bce': 3.0,
  'brain_tumor_rn50_10_brmse': 3.0,
  'brain_tumor_rn50_10_mcc': 4.0,
  'brain_tumor_rn50_250_bce': 3.0,
  'brain_tumor_rn50_250_brmse': 4.0,
  'brain_tumor_rn50_250_mcc': 4.0,
  'brain_tumor_rn50_500_bce': 4.0,
  'brain_tumor_rn50_500_brmse': 4.0,
  'brain_tumor_rn50_500_mcc': 4.0,
  'brain_tumor_rn50_50_bce': 4.0,
  'brain_tumor_rn50_50_brmse': 3.0,
  'brain_tumor_rn50_50_mcc': 3.0,
  'tuberculosis_rn18_100_bce': 3.0,
  'tuberculosis_rn18_100_brmse': 4.0,
  'tuberculosis_rn18_100_mcc': 3.0,
  'tuberculosis_rn18_10_bce': 3.0,
  'tuberculosis_rn18_10_brmse': 3.0,
  'tuberculosis_rn18_10_mcc': 3.0,
  'tuberculosis_rn18_250_bce': 4.0,
  'tuberculosis_rn18_250_brmse': 3.0,
  'tuberculosis_rn18_250_mcc': 3.0,
  'tuberculosis_rn18_500_bce': 3.0,
  'tuberculosis_rn18_500_brmse': 4.0,
  'tuberculosis_rn18_500_mcc': 4.0,
  'tuberculosis_rn18_50_bce': 3.0,
  'tuberculosis_rn18_50_brmse': 4.0,
  'tuberculosis_rn18_50_mcc': 3.0,
  'tuberculosis_rn34_100_bce': 3.0,
  'tuberculosis_rn34_100_brmse': 3.0,
  'tuberculosis_rn34_100_mcc': 3.0,
  'tuberculosis_rn34_10_bce': 3.0,
  'tuberculosis_rn34_10_brmse': 3.0,
  'tuberculosis_rn34_10_mcc': 3.0,
  'tuberculosis_rn34_250_bce': 3.0,
  'tuberculosis_rn34_250_brmse': 3.0,
  'tuberculosis_rn34_250_mcc': 4.0,
  'tuberculosis_rn34_500_bce': 4.0,
  'tuberculosis_rn34_500_brmse': 3.0,
  'tuberculosis_rn34_500_mcc': 4.0,
  'tuberculosis_rn34_50_bce': 3.0,
  'tuberculosis_rn34_50_brmse': 3.0,
  'tuberculosis_rn34_50_mcc': 3.0,
  'tuberculosis_rn50_100_bce': 3.0,
  'tuberculosis_rn50_100_brmse': 3.0,
  'tuberculosis_rn50_100_mcc': 3.0,
  'tuberculosis_rn50_10_bce': 4.0,
  'tuberculosis_rn50_10_brmse': 4.0,
  'tuberculosis_rn50_10_mcc': 3.0,
  'tuberculosis_rn50_250_bce': 3.0,
  'tuberculosis_rn50_250_brmse': 3.0,
  'tuberculosis_rn50_250_mcc': 3.0,
  'tuberculosis_rn50_500_bce': 3.0,
  'tuberculosis_rn50_500_brmse': 4.0,
  'tuberculosis_rn50_500_mcc': 3.0,
  'tuberculosis_rn50_50_bce': 3.0,
  'tuberculosis_rn50_50_brmse': 3.0,
  'tuberculosis_rn50_50_mcc': 3.0}}

seeds = 30

# for i, loader in enumerate(loaders):
#
# for seed in range(10):


def _run(seed, loader, resnet_v, sample_size):

    X, y = loader(model_type=resnet_v, X_y=True)
    dataset_base = loader.__name__.split("load_")[-1]
    dataset = f"{dataset_base}_{resnet_v}_{sample_size}"
    X = StandardScaler().fit_transform(X)

    # Get max depth and width between bce and brmse variants
    key_bce = f"{dataset}_bce"
    key_brmse = f"{dataset}_brmse"
    depth = max(size['depth'][key_bce], size['depth'][key_brmse])
    width = max(size['width'][key_bce], size['width'][key_brmse])

    # Create network for binary classification with sigmoid output
    net = NeuralNetwork(create_network(X.shape[1], width=int(width), depth=int(depth),
                                       num_outputs=1, output_activation=nn.Sigmoid()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-(sample_size/X.shape[0]), seed=seed, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, p_test=0.2, seed=seed, stratify=y_train)

    start = time.time()

    # Train with BCELoss for binary classification (sigmoid already in network)
    net.train_network(X_train, y_train,
                      X_val = X_val, y_val=y_val,
                      X_test=X_test, y_test=y_test,
                      epochs = 1000, batch_size=32,
                      criterion=nn.BCELoss())

    end = time.time()

    for epoch, (train_loss, val_loss, test_loss) in enumerate(zip(net.history["train_loss"], net.history["val_loss"], net.history["test_loss"])):

        logger(f'log/baseline_dropout_evo_{day}.csv',
               generation=epoch,
               timing=0,
               run_info=[dataset , train_loss, val_loss, test_loss],
               seed=seed)

    y_pred = net.forward(X_test).flatten()

    logger(f'log/baseline_dropout_{day}.csv',
           generation=1000,
           timing = end - start,
           run_info = [dataset , binarized_rmse(y_test, y_pred), binarized_bce(y_test, y_pred),  binarized_mcc(y_test, y_pred)],
           seed = seed)




