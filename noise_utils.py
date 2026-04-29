"""
noise_utils.py — Noise injection utilities for robustness experiments.

Two injection modes:
  1. Feature space (post-ResNet): apply_noise() / add_gaussian_noise() etc.
     Features assumed StandardScaler-normalized (mean≈0, std≈1).
  2. Image space (pre-ResNet): apply_image_noise() / apply_image_noise_batch()
     Operates on [0, 1] pixel tensors BEFORE ImageNet normalization and
     ResNet feature extraction.  This is the correct mode for evaluating
     noise robustness — noise propagates through the encoder as in real
     degraded-image scenarios.

Dataset-appropriate noise types:
  - MRI (brain_tumor):                  Gaussian + Rician
  - Chest X-ray (tuberculosis, mc_*):   Gaussian + Poisson

Feature-space functions accept and return either np.ndarray or torch.Tensor.
Image-space functions operate on torch.Tensor [C, H, W] in [0, 1].
"""
import numpy as np
import torch


def _to_np(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    return np.asarray(X, dtype=np.float32)


def _restore(X_noisy, X_orig):
    if isinstance(X_orig, torch.Tensor):
        return torch.tensor(X_noisy, dtype=X_orig.dtype)
    return X_noisy.astype(np.float32)


def add_gaussian_noise(X, sigma=0.1, random_state=None):
    """
    Add Gaussian noise N(0, sigma^2) to feature matrix X.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor, shape (n_samples, n_features)
    sigma : float
        Noise standard deviation. On StandardScaler-normalized features,
        sigma=0.1 (low), 0.5 (medium), 1.0 (high) noise.
    random_state : int or None

    Returns
    -------
    Same type as X with Gaussian noise added.
    """
    rng = np.random.RandomState(random_state)
    X_np = _to_np(X)
    return _restore(X_np + rng.normal(0, sigma, X_np.shape).astype(np.float32), X)


def add_poisson_noise(X, lam=10.0, random_state=None):
    """
    Add Poisson noise appropriate for Chest X-ray features.

    Since StandardScaler output can be negative, the procedure is:
      1. Shift X so all values are positive: X_shifted = X - min(X) + epsilon
      2. Scale and sample: X_poisson = Poisson(lam * X_shifted) / lam
      3. Shift back: X_noisy = X_poisson + offset

    Use lam = 1/sigma to match the unified sigma=[0.1, 0.5, 1.0] scale:
      sigma=0.1 → lam=10 (low noise), sigma=1.0 → lam=1 (high noise).

    Parameters
    ----------
    X : np.ndarray or torch.Tensor, shape (n_samples, n_features)
    lam : float
        Poisson rate parameter. Higher lam = less relative noise.
    random_state : int or None

    Returns
    -------
    Same type as X with Poisson noise added.
    """
    rng = np.random.RandomState(random_state)
    X_np = _to_np(X).astype(np.float64)
    offset = X_np.min() - 1e-6           # shift so all values > 0
    X_shifted = X_np - offset
    scaled = np.clip(lam * X_shifted, 0, None)
    X_noisy = rng.poisson(scaled).astype(np.float64) / lam + offset
    return _restore(X_noisy.astype(np.float32), X)


def add_rician_noise(X, sigma=0.1, random_state=None):
    """
    Add Rician noise appropriate for MRI magnitude image features.

    Formula: K = sqrt((X + n1)^2 + n2^2)
    where n1, n2 ~ N(0, sigma^2) are independent.

    This models the magnitude of a complex Gaussian signal, which is the
    physical source of Rician-distributed noise in MRI magnitude images.
    Note: output is always non-negative; at high sigma this introduces a
    small positive bias (~sigma * sqrt(pi/2)), which is physically correct.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor, shape (n_samples, n_features)
    sigma : float
        Standard deviation of each Gaussian noise component.
    random_state : int or None

    Returns
    -------
    Same type as X with Rician noise applied.
    """
    rng = np.random.RandomState(random_state)
    X_np = _to_np(X)
    n1 = rng.normal(0, sigma, X_np.shape).astype(np.float32)
    n2 = rng.normal(0, sigma, X_np.shape).astype(np.float32)
    return _restore(np.sqrt((X_np + n1) ** 2 + n2 ** 2), X)


def apply_noise(X, noise_type, noise_level, random_state=None):
    """
    Unified interface for applying any supported noise type.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
    noise_type : str — 'gaussian', 'poisson', or 'rician'
    noise_level : float — sigma in [0.1, 0.5, 1.0]
        For Poisson, lam = 1/noise_level is derived internally.
    random_state : int or None

    Returns
    -------
    Noisy version of X in same type.
    """
    if noise_type == 'gaussian':
        return add_gaussian_noise(X, sigma=noise_level, random_state=random_state)
    elif noise_type == 'poisson':
        lam = 1.0 / noise_level if noise_level > 0 else 1.0
        return add_poisson_noise(X, lam=lam, random_state=random_state)
    elif noise_type == 'rician':
        return add_rician_noise(X, sigma=noise_level, random_state=random_state)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'. Supported: gaussian, poisson, rician")


# ---------------------------------------------------------------------------
# Image-space noise  (input: torch.Tensor [C, H, W] in [0, 1])
# ---------------------------------------------------------------------------

def add_gaussian_noise_image(img: torch.Tensor, sigma: float,
                              rng: np.random.RandomState) -> torch.Tensor:
    noise = torch.tensor(rng.normal(0, sigma, img.shape).astype(np.float32),
                         dtype=img.dtype)
    return (img + noise).clamp(0.0, 1.0)


def add_poisson_noise_image(img: torch.Tensor, sigma: float,
                             rng: np.random.RandomState) -> torch.Tensor:
    # lam = 1/sigma: larger sigma → smaller lam → more noise
    lam = max(1.0 / sigma, 1.0)
    scaled = (img.numpy() * lam).clip(0)
    noisy = rng.poisson(scaled).astype(np.float32) / lam
    return torch.tensor(noisy, dtype=img.dtype).clamp(0.0, 1.0)


def add_rician_noise_image(img: torch.Tensor, sigma: float,
                            rng: np.random.RandomState) -> torch.Tensor:
    n1 = rng.normal(0, sigma, img.shape).astype(np.float32)
    n2 = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.sqrt((img.numpy() + n1) ** 2 + n2 ** 2)
    return torch.tensor(noisy, dtype=img.dtype).clamp(0.0, 1.0)


def apply_image_noise(img: torch.Tensor, noise_type: str, noise_level: float,
                      random_state=None) -> torch.Tensor:
    """
    Apply noise to a single image tensor [C, H, W] in [0, 1].

    Parameters
    ----------
    img        : torch.Tensor, shape (C, H, W), values in [0, 1]
    noise_type : str — 'gaussian', 'poisson', or 'rician'
    noise_level: float — sigma in [0.1, 0.5, 1.0]
    random_state: int or None — seed for reproducibility

    Returns
    -------
    torch.Tensor, same shape as img, values clamped to [0, 1]
    """
    rng = np.random.RandomState(random_state)
    if noise_type == 'gaussian':
        return add_gaussian_noise_image(img, noise_level, rng)
    elif noise_type == 'poisson':
        return add_poisson_noise_image(img, noise_level, rng)
    elif noise_type == 'rician':
        return add_rician_noise_image(img, noise_level, rng)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'. Supported: gaussian, poisson, rician")


def apply_image_noise_batch(images: np.ndarray, noise_type: str, noise_level: float,
                             base_seed=None) -> np.ndarray:
    """
    Apply noise independently to each image in a batch.

    Per-image seed is derived as base_seed + i, so results are fully
    reproducible and images receive independent noise draws.

    Parameters
    ----------
    images     : np.ndarray, shape (N, C, H, W), float32 in [0, 1]
    noise_type : str
    noise_level: float
    base_seed  : int or None

    Returns
    -------
    np.ndarray, same shape as images, values in [0, 1]
    """
    out = np.empty_like(images)
    for i in range(len(images)):
        seed_i = None if base_seed is None else base_seed + i
        img_t = torch.from_numpy(images[i])
        out[i] = apply_image_noise(img_t, noise_type, noise_level,
                                   random_state=seed_i).numpy()
    return out


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Noise types per dataset loader (keyed by loader function name)
DATASET_NOISE_TYPES = {
    'load_brain_tumor':     ['gaussian', 'rician'],    # MRI dataset
    'load_tuberculosis':    ['gaussian', 'poisson'],   # Chest X-ray (binary)
    'load_mc_tuberculosis': ['gaussian', 'poisson'],   # Chest X-ray (multiclass)
}

NOISE_LEVELS = [0.1, 0.5, 1.0]          # low / medium / high noise intensity
NOISE_SCENARIOS = ['test_only', 'train_test']
