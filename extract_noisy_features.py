"""
extract_noisy_features.py — Pre-compute noisy ResNet encodings for robustness experiments.

For each dataset × ResNet variant × noise type × noise level:
  1. Load raw images from Kaggle
  2. Apply noise in pixel space ([0,1]) — BEFORE ResNet feature extraction
  3. Save noisy features as CSV to datasets/resnet_features/

Output filenames:
  {dataset}_{resnet_v}_{noise_type}_{noise_level}.csv
  e.g., brain_tumor_rn18_gaussian_0.1.csv

These CSVs are loaded by noise experiment scripts (noise_baseline.py, noise_main.py,
noise_fs_baseline.py) in place of on-the-fly image loading and feature extraction,
making repeated experiments fast.

Row order in each noisy CSV matches the corresponding clean CSV exactly, so the
same train/test split indices can be applied to both.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_resnet_features import load_raw_images, extract_features_from_tensors
from noise_utils import apply_image_noise_batch, DATASET_NOISE_TYPES, NOISE_LEVELS

# Fixed seed for noise — variation across experiments comes from train/test splits
NOISE_SEED = 0

DATASETS = [
    ('load_brain_tumor',     ['rn18', 'rn34', 'rn50']),
    ('load_tuberculosis',    ['rn18', 'rn34', 'rn50']),
    ('load_mc_tuberculosis', ['rn18', 'rn34', 'rn50']),
]

OUTPUT_DIR = Path(__file__).parent / 'datasets' / 'resnet_features'


def _dataset_name(loader_name: str) -> str:
    return loader_name.replace('load_', '')


def extract_and_save_noisy(loader_name: str, resnet_v: str,
                            noise_type: str, noise_level: float,
                            device: str = 'cpu') -> None:
    dataset_name = _dataset_name(loader_name)
    out_path = OUTPUT_DIR / f'{dataset_name}_{resnet_v}_{noise_type}_{noise_level}.csv'

    if out_path.exists():
        print(f'  [skip] {out_path.name} already exists')
        return

    print(f'  Generating {out_path.name} ...')

    # Load raw images [N, 3, 224, 224] in [0, 1]
    images, y = load_raw_images(loader_name)

    # Apply noise to all images in pixel space
    noisy_images = apply_image_noise_batch(images, noise_type, noise_level,
                                           base_seed=NOISE_SEED)

    # Extract ResNet features from noisy images (ImageNet normalization applied inside)
    X = extract_features_from_tensors(noisy_images, resnet_v, device=device)

    # Save as CSV with same format as clean CSVs
    n_features = X.shape[1]
    columns = [f'feature_{i}' for i in range(n_features)] + ['target']
    df = pd.DataFrame(np.column_stack([X, y]), columns=columns)
    df.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}  shape={df.shape}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for loader_name, resnet_versions in DATASETS:
        dataset_name = _dataset_name(loader_name)
        noise_types = DATASET_NOISE_TYPES[loader_name]
        print(f'\n=== {dataset_name} ===')

        for resnet_v in resnet_versions:
            for noise_type in noise_types:
                for noise_level in NOISE_LEVELS:
                    extract_and_save_noisy(loader_name, resnet_v,
                                           noise_type, noise_level, device=device)

    print('\nDone. Noisy CSVs saved to datasets/resnet_features/')


if __name__ == '__main__':
    main()
