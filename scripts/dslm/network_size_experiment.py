import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))
sys.path.insert(0, _here)
LOG_DIR = os.path.join(_here, '..', '..', 'log')

import torch
import datetime
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Any

from DSLM import DSLM
from datasets.data_loader import load_bioav, load_ld50, load_concrete_strength, load_airfoil
from utils.utils import StandardScaler, train_test_split, uniform_random_step_generator
from utils.evaluators import rmse
from population.initializers import initialize_population
from population.selection_algorithms import torunament_selection
from individual.mutation_operators import deflate_mutation, inflate_mutation
from torch import nn, optim

# ── Experiment configuration ────────────────────────────────────────────────

LOADERS = [load_bioav, load_ld50, load_concrete_strength, load_airfoil]

# Network-size configurations: (label, maximum_width, maximum_depth)
# One below the paper default, the paper default itself, and three larger ones.
SIZE_CONFIGS = [
    ('small',   8,  2),   # smaller than paper default (single reference below)
    ('medium',  16, 3),   # paper default
    ('large',   32, 4),
    ('xl',      64, 5),
    ('xxl',    128, 6),
]

SEEDS = 5

# Fixed DSLM / training hyper-parameters (matching paper defaults)
POP_SIZE    = 100
GENERATIONS = 2000
EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 0.001
MS_STEP     = 2       # paper default
PRETRAIN    = 0.5

# Datasets that use balanced inflate/deflate probabilities (paper setting)
_BALANCED_DATASETS = {'airfoil', 'concrete_strength'}


# ── Single experiment ────────────────────────────────────────────────────────

def run_single_experiment(args: Tuple) -> Dict[str, Any]:
    """Run one (dataset × size-config × seed) combination.

    Args:
        args: (loader, config_label, max_width, max_depth, seed,
               per_gen_log_path, experiment_id)

    Returns:
        Dict with final results and metadata.
    """
    loader, config_label, max_width, max_depth, seed, per_gen_log_path, experiment_id = args

    dataset_base = loader.__name__.split('load_')[-1]
    dataset_name = f'{dataset_base}_{config_label}_w{max_width}_d{max_depth}'

    # Dataset-dependent mutation probabilities (paper setting)
    p_im = 0.5 if dataset_base in _BALANCED_DATASETS else 0.3
    p_dm = 1 - p_im

    try:
        X, y = loader(X_y=True)
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed=seed)
        X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, p_test=0.2, seed=seed)

        optimizer = DSLM(
            initializer=initialize_population(
                X_train_nn,
                y_train_nn,
                maximum_width=max_width,
                maximum_depth=max_depth,
                activation_functions=[nn.ReLU()],
                pretrain_part=PRETRAIN,
                X_val=X_val,
                y_val=y_val,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                criterion=nn.MSELoss(),
                optimizer=optim.Adam,
                device='cpu',
            ),
            selector=torunament_selection(2),
            inflate_mutator=inflate_mutation(
                X_train,
                ms_generator=uniform_random_step_generator(0, MS_STEP),
                X_test=X_test,
            ),
            deflate_mutator=deflate_mutation,
            crossover=None,
            p_m=1,
            p_im=p_im,
            p_dm=p_dm,
            p_xo=0,
            pop_size=POP_SIZE,
            seed=seed,
        )

        optimizer.solve(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metric=rmse,
            max_depth=None,
            generations=GENERATIONS,
            elitism=True,
            dataset_name=dataset_name,
            log=1,                       # base per-gen metrics, thread-safe via filelock
            log_path=per_gen_log_path,
            verbose=0,
            n_jobs=-1,
        )

        result = {
            'experiment_id':  experiment_id,
            'dataset':        dataset_base,
            'config_label':   config_label,
            'maximum_width':  max_width,
            'maximum_depth':  max_depth,
            'p_im':           p_im,
            'p_dm':           p_dm,
            'seed':           seed,
            'train_fitness':  optimizer.population.elite.fitness,
            'test_fitness':   optimizer.population.elite.test_fitness,
            'final_size':     optimizer.population.elite.size,
            'status':         'success',
        }

        print(
            f'[{experiment_id}] {dataset_name} seed={seed} '
            f'train={result["train_fitness"]:.4f} test={result["test_fitness"]:.4f}'
        )
        return result

    except Exception as exc:
        print(f'[{experiment_id}] FAILED {dataset_name} seed={seed}: {exc}')
        return {
            'experiment_id': experiment_id,
            'dataset':       dataset_base,
            'config_label':  config_label,
            'maximum_width': max_width,
            'maximum_depth': max_depth,
            'seed':          seed,
            'status':        'failed',
            'error':         str(exc),
        }


# ── Orchestration ────────────────────────────────────────────────────────────

def run_network_size_experiment(
    loaders: List = None,
    size_configs: List[Tuple] = None,
    seeds: int = SEEDS,
    n_jobs: int = None,
) -> List[Dict]:
    """Run the full network-size impact experiment in parallel.

    Args:
        loaders:      List of dataset loader functions (default: all four).
        size_configs: List of (label, max_width, max_depth) tuples.
        seeds:        Number of random seeds per combination.
        n_jobs:       Parallel workers. Defaults to cpu_count() - 1.

    Returns:
        List of result dicts, one per experiment.
    """
    if loaders is None:
        loaders = LOADERS
    if size_configs is None:
        size_configs = SIZE_CONFIGS
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    now = datetime.datetime.now()
    day = now.strftime('%Y%m%d')
    per_gen_log_path = os.path.join(LOG_DIR, f'{day}.csv')
    summary_log_path = os.path.join(LOG_DIR, f'network_size_experiment_{day}.csv')

    # Build experiment list
    experiments = []
    exp_id = 0
    for loader, (label, w, d), seed in itertools.product(
        loaders, size_configs, range(seeds)
    ):
        experiments.append((loader, label, w, d, seed, per_gen_log_path, exp_id))
        exp_id += 1

    total = len(experiments)
    print(f'Network size experiment: {total} runs '
          f'({len(loaders)} datasets x {len(size_configs)} configs x {seeds} seeds)')
    print(f'Per-generation log : {per_gen_log_path}')
    print(f'Summary log        : {summary_log_path}')
    print('=' * 70)

    all_results = []
    completed = 0

    with Pool(processes=n_jobs) as pool:
        for result in pool.imap_unordered(run_single_experiment, experiments):
            all_results.append(result)
            completed += 1

            if completed % 10 == 0 or completed == total:
                df = pd.DataFrame(all_results)
                df.to_csv(summary_log_path, index=False)
                pct = 100 * completed / total
                print(f'Progress: {completed}/{total} ({pct:.1f}%)')

                successful = df[df['status'] == 'success']
                if len(successful) > 0:
                    best = successful['test_fitness'].min()
                    print(f'  Best test fitness so far: {best:.4f}')

    print('=' * 70)
    print(f'Done. Results saved to {summary_log_path}')

    df = pd.DataFrame(all_results)
    successful = df[df['status'] == 'success']
    if len(successful) > 0:
        print(f'\nSuccessful: {len(successful)}/{total}')
        print('\nMean test fitness by config:')
        print(
            successful.groupby(['config_label', 'maximum_width', 'maximum_depth'])
            ['test_fitness'].mean().sort_values()
            .to_string()
        )
        print('\nMean test fitness by dataset x config:')
        print(
            successful.groupby(['dataset', 'config_label'])
            ['test_fitness'].mean().unstack('config_label')
            .to_string()
        )

    return all_results


if __name__ == '__main__':
    run_network_size_experiment()
