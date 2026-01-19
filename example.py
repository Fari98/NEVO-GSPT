"""
Simplified DSLM script using a single dataset.
Demonstrates the core algorithm without multiple datasets or complex logging.
"""

import torch
from torch import nn, optim
from GSPT import GSPT
from datasets.data_loader import load_bioav
from utils.utils import StandardScaler, train_test_split, uniform_random_step_generator, rmse
from population.initializers import initialize_population
from population.selection_algorithms import torunament_selection
from individual.mutation_operators import deflate_mutation, inflate_mutation
from UnifiedModel.UnifiedModel import UnifiedModel


def main():
    # Load and prepare data
    X, y = load_bioav(X_y=True)
    X = StandardScaler().fit_transform(X)

    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed=seed)
    X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, p_test=0.2, seed=seed)

    # Initialize population
    population = initialize_population(
        X_train_nn,
        y_train_nn,
        maximum_width=16,
        maximum_depth=3,
        activation_functions=[nn.ReLU()],
        pretrain_part=0.5,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam,
        device='cpu'
    )

    # Create and run optimizer
    optimizer = GSPT(
        initializer=population,
        selector=torunament_selection(2),
        inflate_mutator=inflate_mutation(X_train, ms_generator=uniform_random_step_generator(0, 2), X_test=X_test),
        deflate_mutator=deflate_mutation,
        crossover=None,
        p_m=1,
        p_im=0.3,
        p_dm=0.7,
        p_xo=0,
        pop_size=100,
        seed=seed,
    )

    optimizer.solve(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metric=rmse,
        max_depth=None,
        generations=100,  # Reduced for faster testing
        elitism=True,
        dataset_name='bioav',
        log=0,  # Disable logging for simplicity
        verbose=1,
        n_jobs=-1
    )

    # Train final model
    final_model = UnifiedModel(optimizer.population.elite.structure)
    final_model.compile()
    history = final_model.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=200,
        batch_size=32
    )

    # Evaluate
    train_pred = final_model.forward(X_train)
    test_pred = final_model.forward(X_test, True)

    train_rmse = rmse(y_train, train_pred.flatten())
    test_rmse = rmse(y_test, test_pred.flatten())

    print(f"\nResults:")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")

    return final_model


if __name__ == "__main__":
    model = main()
