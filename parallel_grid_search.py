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
import pandas as pd
from typing import Dict, List, Any, Tuple
from multiprocessing import Pool, cpu_count, Manager
import os
from functools import partial


# Grid search parameter configuration
PARAM_GRID = {
    # Population parameters
    'pop_size': [50, 100, 200],
    'p_im': [0.3, 0.5, 0.7],  # p_dm will be computed as 1-p_im
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
    'ms_step': [2, 0.2, 0.02, 0.002],
}


def run_single_experiment(args: Tuple[Dict[str, Any], int, Any, str, Any, int, str, int]) -> Dict[str, Any]:
    """
    Run a single experiment with given parameters.
    This function is designed to be called in parallel.

    Args:
        args: Tuple containing (params, seed, loader, resnet_v, metric, size, log_path, experiment_id)

    Returns:
        Dictionary with results
    """
    params, seed, loader, resnet_v, metric, size, log_path, experiment_id = args

    try:
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
        # Note: p_dm = 1 - p_im
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
            p_dm=1 - params['p_im'],  # p_dm is computed from p_im
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
            'experiment_id': experiment_id,
            'dataset': dataset,
            'seed': seed,
            'final_train_fitness': optimizer.population.elite.fitness,
            'final_test_fitness': optimizer.population.elite.test_fitness,
            'final_size': optimizer.population.elite.size,
            'status': 'success'
        }

        # Add all parameters to results, converting activation functions to strings
        for key, value in params.items():
            if key == 'activation_functions':
                # Convert activation function objects to readable string
                results[key] = str([type(act).__name__ for act in value])
            else:
                results[key] = value

        # Add computed p_dm to results
        results['p_dm'] = 1 - params['p_im']

        print(f"✓ Experiment {experiment_id} completed: {dataset} - "
              f"Train: {results['final_train_fitness']:.4f}, "
              f"Test: {results['final_test_fitness']:.4f}")

        return results

    except Exception as e:
        print(f"✗ Experiment {experiment_id} failed: {str(e)}")
        return {
            'experiment_id': experiment_id,
            'status': 'failed',
            'error': str(e),
            'seed': seed,
            'dataset': loader.__name__.split("load_")[-1] if hasattr(loader, '__name__') else 'unknown'
        }


def parallel_grid_search(
    loaders: List = None,
    resnet_versions: List[str] = None,
    metrics: List = None,
    sizes: List[int] = None,
    seeds: int = 10,
    param_grid: Dict[str, List] = None,
    log_path: str = None,
    n_jobs: int = None
):
    """
    Perform parallelized grid search over parameter combinations.

    Args:
        loaders: List of data loading functions
        resnet_versions: List of ResNet versions to test
        metrics: List of evaluation metrics
        sizes: List of training sizes
        seeds: Number of random seeds to run
        param_grid: Dictionary with parameter names and values to search
        log_path: Path to save grid search results
        n_jobs: Number of parallel jobs. If None, uses all available CPUs
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
        log_path = f'log/grid_search_parallel_{day}.csv'
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # Leave one CPU free

    print(f"Starting parallel grid search with {n_jobs} workers")
    print(f"Results will be saved to: {log_path}\n")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    # Build list of all experiments to run
    experiments = []
    experiment_id = 0

    for param_vals in param_combinations:
        params = dict(zip(param_names, param_vals))
        for loader in loaders:
            for resnet_v in resnet_versions:
                for metric in metrics:
                    for size in sizes:
                        for seed in range(seeds):
                            experiments.append((
                                params.copy(),
                                seed,
                                loader,
                                resnet_v,
                                metric,
                                size,
                                log_path,
                                experiment_id
                            ))
                            experiment_id += 1

    total_experiments = len(experiments)
    print(f"Total parameter combinations: {len(param_combinations)}")
    print(f"Total experiments to run: {total_experiments}")
    print(f"{'='*80}\n")

    # Run experiments in parallel
    all_results = []
    completed = 0

    # Use multiprocessing Pool
    with Pool(processes=n_jobs) as pool:
        # Use imap_unordered for better performance and progress tracking
        for result in pool.imap_unordered(run_single_experiment, experiments):
            all_results.append(result)
            completed += 1

            # Save intermediate results every 10 experiments
            if completed % 10 == 0 or completed == total_experiments:
                df = pd.DataFrame(all_results)
                df.to_csv(log_path, index=False)
                print(f"\nProgress: {completed}/{total_experiments} experiments completed "
                      f"({100*completed/total_experiments:.1f}%)")

                # Show current best result
                successful = df[df['status'] == 'success']
                if len(successful) > 0:
                    best_test = successful['final_test_fitness'].min()
                    print(f"Current best test fitness: {best_test:.4f}\n")

    print(f"\n{'='*80}")
    print(f"Grid search complete!")
    print(f"Total experiments run: {len(all_results)}")
    print(f"Results saved to: {log_path}")
    print(f"{'='*80}\n")

    # Print summary statistics
    df = pd.DataFrame(all_results)
    successful = df[df['status'] == 'success']

    if len(successful) > 0:
        print("\nSummary Statistics:")
        print(f"Successful experiments: {len(successful)}/{total_experiments}")
        print(f"Failed experiments: {total_experiments - len(successful)}")
        print(f"\nBest test fitness: {successful['final_test_fitness'].min():.4f}")
        best_idx = successful['final_test_fitness'].idxmin()

        print("\nBest configuration:")
        best_config = successful.loc[best_idx]
        for param in param_names:
            print(f"  {param}: {best_config[param]}")
        print(f"  p_dm: {best_config['p_dm']:.2f} (computed as 1-p_im)")
        print(f"  Dataset: {best_config['dataset']}")
        print(f"  Seed: {best_config['seed']}")
    else:
        print("\nNo successful experiments!")

    return all_results


if __name__ == "__main__":
    # Example 1: Full grid search with parallelization
    # parallel_grid_search(n_jobs=4)

    # Example 2: Smaller grid search for testing
    SMALL_PARAM_GRID = {
        'pop_size': [50, 100],
        'p_im': [0.3, 0.5],  # p_dm = 1 - p_im
        'generations': [500],
        'maximum_width': [8, 16],
        'maximum_depth': [2, 3],
        'pretrain_part': [0.2],
        'epochs': [50, 100],
        'batch_size': [32],
        'learning_rate': [0.001, 0.01],
        'activation_functions': [[nn.ReLU()], [nn.Tanh()], [nn.GELU()]],
        'tournament_size': [2],
        'ms_step': [0.002],
    }

    parallel_grid_search(
        loaders=[load_brain_tumor, load_tuberculosis],
        resnet_versions=['rn18', 'rn50'],
        metrics=[binarized_rmse()],
        sizes=[10, 50],
        seeds=3,
        param_grid=PARAM_GRID,
        n_jobs=-1  # Use 4 parallel workers
    )