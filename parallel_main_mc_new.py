from main_mc_new import _run, loaders, seeds, metrics, resnet_versions, sizes
from joblib import Parallel, delayed
import pandas as pd
import os
from datetime import datetime

now = datetime.now()
# day = now.strftime("%Y%m%d")
day = "20251216"
log_path = f'log/{day}_mc_new.csv'


def get_completed_dataset_seeds():
    """Read log file and extract dataset-seed combinations that reached generation 2000"""
    completed = set()

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path, header=None)
            if len(df) > 0:
                # First column is dataset_name, last column is generation
                # Group by dataset_name and check if max generation >= 2000
                for dataset_name, group in df.groupby(0):
                    max_generation = group.iloc[:, -1].max()
                    if max_generation >= 2000:
                        completed.add(dataset_name)
        except Exception as e:
            print(f"Warning: Could not read log file: {e}")

    return completed


def is_task_completed(loader, version, metric, size):
    """Check if a dataset combination has already been computed"""
    dataset_name = loader.__name__.split("load_")[-1] + '_' + version + '_' + str(size) + '_' + metric.__name__
    return dataset_name in completed_tasks


def generate_tasks():
    """Generate all (seed, loader, version, metric, size) combinations, skipping completed ones"""
    for loader in loaders:
        for metric in metrics:
            for version in resnet_versions:
                for size in sizes:
                    # Skip if this dataset combination is already completed
                    if not is_task_completed(loader, version, metric, size):
                        for seed in range(seeds):
                            yield (seed, loader, version, metric, size)


# Load completed dataset combinations
completed_tasks = get_completed_dataset_seeds()

print(f"Found {len(completed_tasks)} completed dataset combinations")
tasks = list(generate_tasks())
print(f"Total tasks to run: {len(tasks)}")

# Execute all tasks in parallel
_ = Parallel(n_jobs=-1, verbose=10)(
    delayed(_run)(
        seed,
        loader,
        version,
        metric,
        size
    ) for seed, loader, version, metric, size in tasks
)

print(f"\nAll tasks completed!")