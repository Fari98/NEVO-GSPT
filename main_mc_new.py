import torch
from DSLM import DSLM
from datasets.data_loader_resnet import *
from utils.utils import StandardScaler, uniform_random_step_generator
from utils.evaluators import binarized_rmse, binarized_bce, binarized_mcc, cross_entropy
from population.initializers import initialize_population
from population.selection_algorithms import torunament_selection
from individual.mutation_operators import deflate_mutation, inflate_mutation
from torch import nn
from torch import optim
import datetime
from sklearn.model_selection import train_test_split
from UnifiedModel.UnifiedModel import UnifiedModel
from utils.info import base_logger

now = datetime.datetime.now()
# day = now.strftime("%Y%m%d")
day = "20251216"

loaders = [
           load_mc_tuberculosis
]

resnet_versions = ['rn18', 'rn34', 'rn50']
metrics = [
        cross_entropy
           ]
sizes = [10, 50, 100, 250, 500]
# sizes = [500]
# m_steps = [0.2, 0.002]
seeds = 5


def _run( seed, loader, resnet_v, metric, size):

    X, y = loader(model_type= resnet_v, X_y = True)
    dataset = loader.__name__.split("load_")[-1] + '_' + resnet_v + '_' + str(size) + '_' + metric.__name__
    X = StandardScaler().fit_transform(X)

    X_train,  X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state = seed, stratify=y)

    X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = seed, stratify=y_train)

    X_train, X_test, y_train, y_test, X_train_nn, X_val, y_train_nn, y_val = (
        torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(y_train).squeeze(),
        torch.Tensor(y_test).squeeze(), torch.Tensor(X_train_nn), torch.Tensor(X_val),
        torch.Tensor(y_train_nn).squeeze(), torch.Tensor(y_val).squeeze())

    optimizer = DSLM(initializer = initialize_population(X_train_nn,
                                                         y_train_nn,
                                                         maximum_width=16,
                                                         maximum_depth=3,
                                                         n_classes = y.unique().shape[0],
                                                         activation_functions=[nn.ReLU()],
                                                         pretrain_part=0.5,
                                                         X_val=X_val, y_val=y_val,
                                                         epochs=100, batch_size=32, learning_rate=0.001,
                                                         criterion=nn.MSELoss(),
                                                         optimizer=optim.Adam,
                                                         device='cpu'
                                                         ),
                    selector = torunament_selection(2),
                    inflate_mutator = inflate_mutation(X_train,
                                                       ms_generator=uniform_random_step_generator(0, 0.2),
                                                       X_test=X_test,
                                                       n_classes=y.unique().shape[0]),
                    deflate_mutator = deflate_mutation,
                    crossover=None,
                    p_m = 1,
                    p_im=0.3,
                    p_dm=0.7,
                    p_xo=0,
                    pop_size=100,
                    seed=seed,
                    normalize_semantics=True  # NEW: Enable normalized semantics
                )

    optimizer.solve(X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                multiclass=True,
                metric = metric,
                max_depth = None,
                generations=2000,
                elitism=True,
                dataset_name=dataset,
                log=3,
                log_path = f'log/{day}_mc_new_0_2.csv' , # NEW: Different log file name
                verbose=1,
                n_jobs = -1,
                n_classes = y.unique().shape[0]
            )

    # final_model = UnifiedModel(optimizer.population.elite.structure)
    # final_model.compile()
    # history = final_model.fit(X_train, y_train,
    #                 X_val=X_test, y_val=y_test,
    #                 epochs=1000, batch_size=32 if dataset != 'ld50' else 4)
    #
    # for epoch, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"])):
    #     base_logger(path = f'log/final_model_evo_{day}.csv',
    #                row=[dataset,
    #                      seed,
    #                      epoch,
    #                      train_loss,
    #                      val_loss])
    #
    # train_pred = final_model.forward(X_train)
    # test_pred = final_model.forward(X_test, True)
    #
    # base_logger(path = f'log/final_model_trained_{day}.csv',
    #             row = [dataset,
    #                     seed,
    #                     rmse(y_train, train_pred.flatten()),
    #                     rmse(y_test, test_pred.flatten())]
    # )



    return 'done'


if __name__ == '__main__':
    # Run experiments
    for loader in loaders:
        for resnet_v in resnet_versions:
            for metric in metrics:
                for size in sizes:
                    for seed in range(seeds):
                        print(f"\nRunning: {loader.__name__} - {resnet_v} - {metric.__name__} - size={size} - seed={seed}")
                        _run(seed, loader, resnet_v, metric, size)
                        print(f"Completed: {loader.__name__} - {resnet_v} - {metric.__name__} - size={size} - seed={seed}")