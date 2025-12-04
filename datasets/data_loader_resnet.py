"""
Load ResNet-extracted feature datasets from CSV files.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path


def load_dataset(dataset_name, model_type='rn18', X_y=True):
    """
    Load ResNet-extracted feature dataset from CSV file.

    Args:
        dataset_name (str): Name of the dataset.
            Options: 'brain_tumor', 'tuberculosis', 'mc_tuberculosis'
        model_type (str): ResNet model type for feature extraction.
            Options: 'rn18' (ResNet18), 'rn34' (ResNet34), 'rn50' (ResNet50)
            Default: 'rn18'
        X_y (bool): If True, return (X, y) tuple. If False, return DataFrame.
            Default: True

    Returns:
        If X_y=True: (X, y) where X is torch.Tensor of shape (n_samples, n_features) and y is torch.Tensor
            - brain_tumor and tuberculosis: n_features=512
            - mc_tuberculosis: n_features=512
            - rn50 variants: n_features=2048
        If X_y=False: DataFrame with features and target

    Raises:
        FileNotFoundError: If the requested CSV file does not exist
        ValueError: If dataset_name or model_type is invalid
    """
    valid_datasets = {'brain_tumor', 'tuberculosis', 'mc_tuberculosis'}
    valid_models = {'rn18', 'rn34', 'rn50'}

    if dataset_name not in valid_datasets:
        raise ValueError(f"dataset_name must be one of {valid_datasets}, got '{dataset_name}'")
    if model_type not in valid_models:
        raise ValueError(f"model_type must be one of {valid_models}, got '{model_type}'")

    csv_path = Path(__file__).parent / 'resnet_features' / f'{dataset_name}_{model_type}.csv'

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if X_y:
        X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
        return X, y
    else:
        return df


def load_brain_tumor(model_type='rn18', X_y=True):
    """
    Load brain tumor dataset (ResNet-extracted features).

    Args:
        model_type (str): ResNet model type. Options: 'rn18', 'rn34', 'rn50'. Default: 'rn18'
        X_y (bool): If True, return (X, y). If False, return DataFrame. Default: True

    Returns:
        If X_y=True: (X, y) where X has shape (n_samples, 512) and y is binary target
        If X_y=False: DataFrame with features and target
    """
    return load_dataset('brain_tumor', model_type=model_type, X_y=X_y)


def load_tuberculosis(model_type='rn18', X_y=True):
    """
    Load tuberculosis dataset (ResNet-extracted features, binary classification).

    Args:
        model_type (str): ResNet model type. Options: 'rn18', 'rn34', 'rn50'. Default: 'rn18'
        X_y (bool): If True, return (X, y). If False, return DataFrame. Default: True

    Returns:
        If X_y=True: (X, y) where X has shape (n_samples, 512) and y is binary target
        If X_y=False: DataFrame with features and target
    """
    return load_dataset('tuberculosis', model_type=model_type, X_y=X_y)


def load_mc_tuberculosis(model_type='rn18', X_y=True):
    """
    Load tuberculosis dataset (ResNet-extracted features, multiclass classification).

    Classes:
        0: Normal
        1: Localized TB
        2: Bilateral/Disseminated TB
        3: With complications

    Args:
        model_type (str): ResNet model type. Options: 'rn18', 'rn34', 'rn50'. Default: 'rn18'
        X_y (bool): If True, return (X, y). If False, return DataFrame. Default: True

    Returns:
        If X_y=True: (X, y) where X has shape (n_samples, 512) and y is multiclass target
        If X_y=False: DataFrame with features and target
    """
    return load_dataset('mc_tuberculosis', model_type=model_type, X_y=X_y)


