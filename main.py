import torch
from DSLM import DSLM
from datasets.data_loader import *
from utils.utils import StandardScaler, train_test_split, uniform_random_step_generator, rmse, mse
from population.initializers import initialize_population
from population.selection_algorithms import torunament_selection
from individual.mutation_operators import deflate_mutation, inflate_mutation
from torch import nn
from torch import optim
import datetime
from UnifiedModel.UnifiedModel import UnifiedModel
from utils.info import base_logger

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

loaders = [
           load_bioav,
           load_ld50,
           load_concrete_strength,
           load_airfoil
]
seeds = 30
# for loader in [load_bioav, load_ld50, load_concrete_strength, load_airfoil]:
#
# for seed in range(10):

def _run( seed, loader):

    X, y = loader(X_y = True)
    dataset = loader.__name__.split("load_")[-1]
    X = StandardScaler().fit_transform(X)

    X_train,  X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed = seed)

    X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, p_test=0.2, seed = seed)


    optimizer = DSLM(initializer = initialize_population(X_train_nn,
                                                         y_train_nn,
                                                         maximum_width=16,
                                                         maximum_depth=3,
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
                                                       ms_generator=uniform_random_step_generator(0, 2), #todo set as median of y -> overfits(input is scaled) torch.median(y_train).item()
                                                       X_test=X_test),
                    deflate_mutator = deflate_mutation,
                    crossover=None,
                    p_m = 1,
                    p_im=0.5 if dataset in ['airfoil', 'concrete_strength'] else 0.3,
                    p_dm=0.5 if dataset in ['airfoil', 'concrete_strength'] else 0.7,
                    p_xo=0,
                    pop_size=100,
                    seed=seed,
                )

    optimizer.solve(X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                metric = rmse,
                max_depth = None,
                generations=2000,
                elitism=True,
                dataset_name=dataset,
                log=1,
                log_path = f'log/{day}.csv' , #f'log/{day}_nopretrain.csv'
                verbose=1,
                n_jobs = -1
            )

    final_model = UnifiedModel(optimizer.population.elite.structure)
    final_model.compile()
    history = final_model.fit(X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=1000, batch_size=32 if dataset != 'ld50' else 4)

    for epoch, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"])):
        base_logger(path = f'log/final_model_evo_{day}.csv',
                   row=[dataset,
                         seed,
                         epoch,
                         train_loss,
                         val_loss])

    train_pred = final_model.forward(X_train)
    test_pred = final_model.forward(X_test, True)

    base_logger(path = f'log/final_model_trained_{day}.csv',
                row = [dataset,
                        seed,
                        rmse(y_train, train_pred.flatten()),
                        rmse(y_test, test_pred.flatten())]
    )



    return 'done'