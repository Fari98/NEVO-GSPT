import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))
sys.path.insert(0, _here)
LOG_DIR = os.path.join(_here, '..', '..', 'log')

import torch
from DSLM import DSLM
from datasets.data_loader_resnet import *
from utils.utils import StandardScaler, uniform_random_step_generator
from sklearn.model_selection import train_test_split
from utils.evaluators import binarized_rmse, binarized_bce, binarized_mcc
from population.initializers import initialize_population
from population.selection_algorithms import torunament_selection
from individual.mutation_operators import deflate_mutation, inflate_mutation
from torch import nn
from torch import optim
import datetime
import itertools
from utils.info import base_logger
import pandas as pd
from typing import Dict, List, Any


# Grid search parameter configuration
PARAM_GRID = {
    # Population parameters
    'pop_size': [50, 100, 200],
    'p_im': [0.3, 0.5, 0.7],
    # 'p_dm': [0.3, 0.5, 0.7],
    'generations': [2000],

    # Network initialization parameters
    'maximum_width': [8, 16, 32],
    'maximum_depth': [2, 3, 4],
    'pretrain_part': [0.2],
    'epochs': [50, 100, 200],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.0001, 0.001, 0.01],
    'activation_functions': [
        [nn.ReLU()],
        [nn.Tanh()],
        [nn.Sigmoid()],
        [nn.LeakyReLU()],
        [nn.ELU()],
        [nn.GELU()],
    ],

    # Selection parameters
    'tournament_size': [2, 3, 4],

    # Mutation parameters
    'ms_step': [2, 0.2, 0.02, 0.002, 0.002],
}


def run_experiment(params: Dict[str, Any], seed: int, loader, resnet_v: str,
                   metric, size: int, log_path: str) -> Dict[str, Any]:
    """
    Run a single experiment with given parameters.

    Args:
        params: Dictionary of parameters for this run
        seed: Random seed
        loader: Data loading function
        resnet_v: ResNet version
        metric: Evaluation metric
        size: Training size
        log_path: Path to log file

    Returns:
        Dictionary with results
    """
    # Load and prepare data
    X, y = loader(model_type=resnet_v, X_y=True)
    dataset = loader.__name__.split("load_")[-1] + '_' + resnet_v + '_' + str(size) + '_' + metric.__name__
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=size, random_state=seed, stratify=y
    )

    X_train_nn, X_val, y_train_nn, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    # Convert to tensors
    X_train, X_test, y_train, y_test, X_train_nn, X_val, y_train_nn, y_val = (
        torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(y_train).squeeze(),
        torch.Tensor(y_test).squeeze(), torch.Tensor(X_train_nn), torch.Tensor(X_val),
        torch.Tensor(y_train_nn).squeeze(), torch.Tensor(y_val).squeeze()
    )

    # Initialize DSLM with grid search parameters
    optimizer = DSLM(
        initializer=initialize_population(
            X_train_nn,
            y_train_nn,
            maximum_width=params['maximum_width'],
            maximum_depth=params['maximum_depth'],
            activation_functions=params['activation_functions'],
            pretrain_part=params['pretrain_part'],
            X_val=X_val,
            y_val=y_val,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            criterion=nn.MSELoss(),
            optimizer=optim.Adam,
            device='cpu'
        ),
        selector=torunament_selection(params['tournament_size']),
        inflate_mutator=inflate_mutation(
            X_train,
            ms_generator=uniform_random_step_generator(0, params['ms_step']),
            X_test=X_test
        ),
        deflate_mutator=deflate_mutation,
        crossover=None,
        p_m=1,
        p_im=params['p_im'],
        p_dm=1-params['p_im'],
        p_xo=0,
        pop_size=params['pop_size'],
        seed=seed,
    )

    # Run optimization
    optimizer.solve(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metric=metric,
        max_depth=None,
        generations=params['generations'],
        elitism=True,
        dataset_name=dataset,
        log=0,  # Disable individual logging
        log_path=None,
        verbose=0,  # Suppress console output during grid search
        n_jobs=-1
    )

    # Collect results
    results = {
        'dataset': dataset,
        'seed': seed,
        'final_train_fitness': optimizer.population.elite.fitness,
        'final_test_fitness': optimizer.population.elite.test_fitness,
        'final_size': optimizer.population.elite.size,
    }

    # Add all parameters to results, converting activation functions to strings
    for key, value in params.items():
        if key == 'activation_functions':
            # Convert activation function objects to readable string
            results[key] = str([type(act).__name__ for act in value])
        else:
            results[key] = value

    return results


def grid_search(
    loaders: List = None,
    resnet_versions: List[str] = None,
    metrics: List = None,
    sizes: List[int] = None,
    seeds: int = 10,
    param_grid: Dict[str, List] = None,
    log_path: str = None
):
    """
    Perform grid search over parameter combinations.

    Args:
        loaders: List of data loading functions
        resnet_versions: List of ResNet versions to test
        metrics: List of evaluation metrics
        sizes: List of training sizes
        seeds: Number of random seeds to run
        param_grid: Dictionary with parameter names and values to search
        log_path: Path to save grid search results
    """
    # Use defaults if not provided
    if loaders is None:
        loaders = [load_brain_tumor, load_tuberculosis]
    if resnet_versions is None:
        resnet_versions = ['rn18', 'rn34', 'rn50']
    if metrics is None:
        metrics = [binarized_rmse(), binarized_bce()]
    if sizes is None:
        sizes = [10, 50, 100, 250, 500]
    if param_grid is None:
        param_grid = PARAM_GRID
    if log_path is None:
        now = datetime.datetime.now()
        day = now.strftime("%Y%m%d")
        log_path = os.path.join(LOG_DIR, f'grid_search_{day}.csv')

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    print(f"Total parameter combinations: {len(param_combinations)}")
    print(f"Total experiments: {len(param_combinations) * len(loaders) * len(resnet_versions) * len(metrics) * len(sizes) * seeds}")
    print(f"Results will be saved to: {log_path}")
    print("\nStarting grid search...\n")

    all_results = []
    experiment_count = 0

    # Iterate over all combinations
    for combo_idx, param_values in enumerate(param_combinations, 1):
        # Create parameter dictionary for this combination
        params = dict(zip(param_names, param_values))

        print(f"\n{'='*80}")
        print(f"Parameter Combination {combo_idx}/{len(param_combinations)}")
        print(f"{'='*80}")
        print("Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()

        # Run experiments for this parameter combination
        for loader in loaders:
            for resnet_v in resnet_versions:
                for metric in metrics:
                    for size in sizes:
                        for seed in range(seeds):
                            experiment_count += 1

                            dataset_name = loader.__name__.split("load_")[-1]
                            print(f"Experiment {experiment_count}: "
                                  f"{dataset_name}_{resnet_v}_{size}_{metric.__name__} "
                                  f"(seed {seed})")

                            try:
                                result = run_experiment(
                                    params=params,
                                    seed=seed,
                                    loader=loader,
                                    resnet_v=resnet_v,
                                    metric=metric,
                                    size=size,
                                    log_path=log_path
                                )
                                all_results.append(result)

                                # Save intermediate results
                                df = pd.DataFrame(all_results)
                                df.to_csv(log_path, index=False)

                                print(f"  ✓ Train: {result['final_train_fitness']:.4f}, "
                                      f"Test: {result['final_test_fitness']:.4f}, "
                                      f"Size: {result['final_size']}")

                            except Exception as e:
                                print(f"  ✗ Error: {str(e)}")
                                continue

    print(f"\n{'='*80}")
    print(f"Grid search complete!")
    print(f"Total experiments run: {len(all_results)}")
    print(f"Results saved to: {log_path}")
    print(f"{'='*80}\n")

    # Print summary statistics
    if all_results:
        df = pd.DataFrame(all_results)
        print("\nSummary Statistics:")
        print(f"Best test fitness: {df['final_test_fitness'].min():.4f}")
        best_idx = df['final_test_fitness'].idxmin()
        print("\nBest configuration:")
        for col in param_names:
            print(f"  {col}: {df.loc[best_idx, col]}")

    return all_results


if __name__ == "__main__":
    # Example 1: Full grid search (WARNING: This will take a very long time!)
    # grid_search()

    # Example 2: Smaller grid search for testing
    SMALL_PARAM_GRID = {
        'pop_size': [50, 100],
        'p_im': [0.3, 0.5],
        'p_dm': [0.5, 0.7],
        'generations': [500, 1000],
        'maximum_width': [8, 16],
        'maximum_depth': [2, 3],
        'pretrain_part': [0.3, 0.5],
        'epochs': [50, 100],
        'batch_size': [32],
        'learning_rate': [0.001, 0.01],
        'activation_functions': [[nn.ReLU()], [nn.Tanh()]],
        'tournament_size': [2],
        'ms_step': [0.002],
    }

    grid_search(
        loaders=[load_brain_tumor, load_tuberculosis()],
        resnet_versions=['rn18', 'rn50'],
        metrics=[binarized_rmse()],
        sizes=[10, 50],
        seeds=3,
        param_grid=PARAM_GRID
    )