from baseline_vit import _run, sample_sizes, seeds
from joblib import Parallel, delayed
import pandas as pd
import os
from datetime import datetime
import kagglehub

now = datetime.now()
day = now.strftime("%Y%m%d")
log_path = f'log/baseline_vit_{day}.csv'

# Dataset names
datasets = ['brain_tumor', 'tuberculosis']


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
                    if max_generation >= 1000:
                        completed.add(dataset_name)
        except Exception as e:
            print(f"Warning: Could not read log file: {e}")

    return completed


def is_task_completed(dataset_name, size):
    """Check if a dataset combination has already been computed"""
    full_name = f"{dataset_name}_vit_{size}"
    return full_name in completed_tasks


def generate_tasks(dataset_paths):
    """Generate all (seed, dataset_name, size, dataset_path) combinations, skipping completed ones"""
    for dataset_name in datasets:
        for size in sample_sizes:
            # Skip if this dataset combination is already completed
            if not is_task_completed(dataset_name, size):
                for seed in range(seeds):
                    yield (seed, dataset_name, size, dataset_paths[dataset_name])


# Load completed dataset combinations
completed_tasks = get_completed_dataset_seeds()

print(f"Found {len(completed_tasks)} completed tasks")
print(f"Completed: {completed_tasks}")

# Download datasets once before parallel execution
print("\nDownloading datasets...")
dataset_paths = {}
print("Downloading brain_tumor dataset...")
dataset_paths['brain_tumor'] = kagglehub.dataset_download("ahmedhamada0/brain-tumor-detection")
print(f"  -> {dataset_paths['brain_tumor']}")

print("Downloading tuberculosis dataset...")
dataset_paths['tuberculosis'] = kagglehub.dataset_download("raddar/tuberculosis-chest-xrays-shenzhen")
print(f"  -> {dataset_paths['tuberculosis']}")

print("\nStarting parallel execution...")

# Execute all tasks in parallel
_ = Parallel(n_jobs=-1)(
    delayed(_run)(
        seed,
        dataset_name,
        size,
        dataset_path=dataset_path,
        freeze_backbone=True  # Can change to False to fine-tune entire model
    ) for seed, dataset_name, size, dataset_path in generate_tasks(dataset_paths)
)

print("\nAll experiments completed!")