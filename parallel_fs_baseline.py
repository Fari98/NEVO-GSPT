from fs_baseline import _run, loaders, seeds, resnet_versions, sizes
from joblib import Parallel, delayed
import pandas as pd
import os
from datetime import datetime

now = datetime.now()
day = now.strftime("%Y%m%d")
log_path = f'log/{day}_fs_baseline.csv'


def get_completed_tasks():
    """Read log file and extract completed dataset-seed combinations"""
    completed = set()

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            if len(df) > 0:
                # Create unique identifier for each completed task
                for _, row in df.iterrows():
                    task_id = (row['dataset'], row['seed'])
                    completed.add(task_id)
        except Exception as e:
            print(f"Warning: Could not read log file: {e}")

    return completed


def is_task_completed(loader, resnet_v, size, seed):
    """Check if a task has already been completed"""
    dataset_name = loader.__name__.split("load_")[-1] + '_' + resnet_v + '_' + str(size)
    task_id = (dataset_name, seed)
    return task_id in completed_tasks


def generate_tasks():
    """Generate all (seed, loader, resnet_v, size) combinations, skipping completed ones"""
    for loader in loaders:
        for resnet_v in resnet_versions:
            for size in sizes:
                for seed in range(seeds):
                    # Skip if this task is already completed
                    if not is_task_completed(loader, resnet_v, size, seed):
                        yield (seed, loader, resnet_v, size)


# Load completed tasks
completed_tasks = get_completed_tasks()

print(f"Found {len(completed_tasks)} completed tasks")
tasks = list(generate_tasks())
print(f"Total tasks to run: {len(tasks)}")

# Execute all tasks in parallel
_ = Parallel(n_jobs=-1, verbose=10)(
    delayed(_run)(
        seed,
        loader,
        resnet_v,
        size
    ) for seed, loader, resnet_v, size in tasks
)

print(f"\nAll tasks completed!")