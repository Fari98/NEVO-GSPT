"""
parallel_noise_baseline.py — Parallel NN baseline noise robustness experiments.

Runs noise_baseline._run() in parallel via joblib. Supports resumption.
"""
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))
sys.path.insert(0, _here)

from noise_baseline import _run, loaders, resnet_versions, sample_sizes, seeds
from noise_utils import DATASET_NOISE_TYPES, NOISE_LEVELS, NOISE_SCENARIOS
from joblib import Parallel, delayed
import pandas as pd
from datetime import datetime

now = datetime.now()
day = now.strftime("%Y%m%d")
LOG_DIR = os.path.join(_here, '..', '..', 'log')
log_path = os.path.join(LOG_DIR, f'noise_baseline_{day}.csv')


def get_completed():
    """Return set of (dataset, noise_type, noise_level, noise_scenario) already logged."""
    if not os.path.exists(log_path):
        return set()
    try:
        df = pd.read_csv(log_path, header=None)
        # Columns: dataset(0), noise_type(1), noise_level(2), noise_scenario(3), ...
        return set(zip(df[0], df[1], df[2], df[3]))
    except Exception as e:
        print(f"Warning: could not read log: {e}")
        return set()


def generate_tasks():
    completed = get_completed()
    for loader in loaders:
        for resnet_v in resnet_versions:
            for sample_size in sample_sizes:
                for noise_type in DATASET_NOISE_TYPES[loader.__name__]:
                    for noise_level in NOISE_LEVELS:
                        for noise_scenario in NOISE_SCENARIOS:
                            dataset_base = loader.__name__.split("load_")[-1]
                            dataset = f"{dataset_base}_{resnet_v}_{sample_size}"
                            key = (dataset, noise_type, noise_level, noise_scenario)
                            if key not in completed:
                                for seed in range(seeds):
                                    yield (seed, loader, resnet_v, sample_size,
                                           noise_type, noise_level, noise_scenario)


_ = Parallel(n_jobs=-1)(
    delayed(_run)(seed, loader, resnet_v, sample_size, noise_type, noise_level, noise_scenario)
    for seed, loader, resnet_v, sample_size, noise_type, noise_level, noise_scenario
    in generate_tasks()
)
