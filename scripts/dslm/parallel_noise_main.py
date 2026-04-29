"""
parallel_noise_main.py — Parallel DSLM noise robustness experiments.

Runs noise_main._run() in parallel via joblib. Supports resumption:
completed runs are detected by their dataset string in the log file.
"""
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))
sys.path.insert(0, _here)

from noise_main import _run, loaders, resnet_versions, metrics, sizes, seeds
from noise_utils import DATASET_NOISE_TYPES, NOISE_LEVELS, NOISE_SCENARIOS
from joblib import Parallel, delayed
import pandas as pd
from datetime import datetime

now = datetime.now()
day = now.strftime("%Y%m%d")
LOG_DIR = os.path.join(_here, '..', '..', 'log')
log_path = os.path.join(LOG_DIR, f'noise_dslm_{day}.csv')


def get_completed():
    """Return set of dataset strings already present in the log."""
    if not os.path.exists(log_path):
        return set()
    try:
        df = pd.read_csv(log_path, header=None)
        return set(df[0].tolist())
    except Exception as e:
        print(f"Warning: could not read log: {e}")
        return set()


def generate_tasks():
    completed = get_completed()
    for loader in loaders:
        for resnet_v in resnet_versions:
            for metric in metrics:
                for size in sizes:
                    for noise_type in DATASET_NOISE_TYPES[loader.__name__]:
                        for noise_level in NOISE_LEVELS:
                            for noise_scenario in NOISE_SCENARIOS:
                                dataset_base = loader.__name__.split("load_")[-1]
                                key = (f"{dataset_base}_{resnet_v}_{size}_{metric.__name__}"
                                       f"_{noise_type}_{noise_level}_{noise_scenario}")
                                if key not in completed:
                                    for seed in range(seeds):
                                        yield (seed, loader, resnet_v, metric, size,
                                               noise_type, noise_level, noise_scenario)


_ = Parallel(n_jobs=-1)(
    delayed(_run)(seed, loader, resnet_v, metric, size, noise_type, noise_level, noise_scenario)
    for seed, loader, resnet_v, metric, size, noise_type, noise_level, noise_scenario
    in generate_tasks()
)
