from baseline import _run, loaders, seeds, resnet_versions, sample_sizes
from joblib import Parallel, delayed
import pandas as pd
import os
from datetime import datetime

now = datetime.now()
day = now.strftime("%Y%m%d")
log_path = f'log/baseline_dropout_{day}.csv'


def get_completed_dataset_seeds():
    """Read log file and extract dataset-seed combinations that are completed"""
    completed = set()

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path, header=None)
            if len(df) > 0:
                # First column is dataset_name, last column is generation
                # Check if generation == 1000 (final entry)
                for dataset_name, group in df.groupby(0):
                    max_generation = group.iloc[:, -1].max()
                    if max_generation >= 2000:
                        completed.add(dataset_name)
        except Exception as e:
            print(f"Warning: Could not read log file: {e}")

    return completed


def is_task_completed(loader, version, size):
    """Check if a dataset combination has already been computed"""
    dataset_name = loader.__name__.split("load_")[-1] + '_' + version + '_' + str(size)
    return dataset_name in completed_tasks


def generate_tasks():
    """Generate all (seed, loader, version, size) combinations, skipping completed ones"""
    for loader in loaders:
        for version in resnet_versions:
            for size in sample_sizes:
                # Skip if this dataset combination is already completed
                if not is_task_completed(loader, version, size):
                    for seed in range(10,30): #todo range(10, 30)








                        yield (seed, loader, version, size)


# Load completed dataset combinations
completed_tasks = get_completed_dataset_seeds()

# Execute all tasks in parallel
_ = Parallel(n_jobs=-1)(
    delayed(_run)(
        seed,
        loader,
        version,
        size
    ) for seed, loader, version, size in generate_tasks()
)

