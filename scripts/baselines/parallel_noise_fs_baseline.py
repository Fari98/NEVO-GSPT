"""
parallel_noise_fs_baseline.py — Parallel few-shot baseline noise robustness.

Runs noise_fs_baseline._run() in parallel via joblib. Supports resumption.
"""
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))
sys.path.insert(0, _here)

from noise_fs_baseline import _run, loaders, resnet_versions, sizes, seeds
from noise_utils import DATASET_NOISE_TYPES, NOISE_LEVELS, NOISE_SCENARIOS
from joblib import Parallel, delayed
import pandas as pd
from datetime import datetime

now = datetime.now()
day = now.strftime("%Y%m%d")
LOG_DIR = os.path.join(_here, '..', '..', 'log')
log_path = os.path.join(LOG_DIR, f'noise_fs_baseline_{day}.csv')


def get_completed():
    """Return set of (dataset, seed, noise_type, noise_level, noise_scenario) already logged."""
    if not os.path.exists(log_path):
        return set()
    try:
        df = pd.read_csv(log_path)
        return set(zip(df['dataset'], df['seed'], df['noise_type'],
                       df['noise_level'], df['noise_scenario']))
    except Exception as e:
        print(f"Warning: could not read log: {e}")
        return set()


def generate_tasks():
    completed = get_completed()
    for loader in loaders:
        for resnet_v in resnet_versions:
            for size in sizes:
                for noise_type in DATASET_NOISE_TYPES[loader.__name__]:
                    for noise_level in NOISE_LEVELS:
                        for noise_scenario in NOISE_SCENARIOS:
                            dataset_base = loader.__name__.split("load_")[-1]
                            dataset = f"{dataset_base}_{resnet_v}_{size}"
                            for seed in range(seeds):
                                key = (dataset, seed, noise_type, noise_level, noise_scenario)
                                if key not in completed:
                                    yield (seed, loader, resnet_v, size,
                                           noise_type, noise_level, noise_scenario)


_ = Parallel(n_jobs=-1)(
    delayed(_run)(seed, loader, resnet_v, size, noise_type, noise_level, noise_scenario)
    for seed, loader, resnet_v, size, noise_type, noise_level, noise_scenario
    in generate_tasks()
)
