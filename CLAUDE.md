# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DSLM** (Dimensionality Scaling Language Model) is a semantic-based evolutionary algorithm for neural architecture search. It evolves populations of neural network structures by inflating (adding) and deflating (removing) neural network **Blocks** using semantic-aware genetic programming. The primary application domain is classification on pre-computed ResNet image features.

## How to Run

There is no build/install step. Scripts are run directly from the repo root. All scripts prepend `sys.path` with the repo root so imports work without installing the package.

**Binary classification (brain tumor, tuberculosis):**
```bash
python scripts/dslm/main.py
```

**Multiclass classification:**
```bash
python scripts/dslm/main_mc.py
# or optimized variant:
python scripts/dslm/main_mc_new.py
```

**Baselines:**
```bash
python scripts/baselines/baseline.py          # standard NN
python scripts/baselines/fs_baseline.py       # feature selection baseline
python scripts/baselines/noise_baseline.py    # noise robustness
```

**Grid search:**
```bash
python scripts/grid_search/grid_search.py
python scripts/grid_search/parallel_grid_search.py
```

**Parallel execution** (runs all dataset/metric/size combos concurrently, with checkpoint skipping):
```bash
python scripts/dslm/parallel_main.py
python scripts/dslm/parallel_main_mc.py
```

No test suite exists — validation is done by inspecting CSV logs in `log/`.

## Architecture

### Core Evolutionary Loop (`DSLM.py`)

The `DSLM` class orchestrates the full evolutionary algorithm:
1. **Initialization**: `population/initializers.py` creates random `NeuralNetwork` objects, trains them on a validation split, and wraps them as `Individual` objects.
2. **Selection**: Tournament selection (`population/population.py`) picks parents from the population.
3. **Mutation**: `individual/mutation_operators.py` applies either `inflate_mutation` (adds a new Block) or `deflate_mutation` (removes a random Block).
4. **Evaluation**: Fitness computed via evaluators in `utils/evaluators.py`.
5. **Elitism**: Optionally preserves best individual across generations.

### Individual & Semantics

An `Individual` (`individual/individual.py`) holds:
- A **structure**: ordered list of `Block` objects.
- **train/test semantics**: cached prediction tensors. For binary: shape `[batch, n_blocks]`. For multiclass: shape `[batch, n_blocks, n_classes]`.

Semantics are updated incrementally — inflation concatenates new block output, deflation slices it out — avoiding full recomputation.

### Block (`block/block.py`)

A `Block` is a PyTorch `nn.Module` that:
- Takes the **hidden layer outputs** of the previous block (or initial NeuralNetwork) as input.
- Produces new predictions scaled by a mutation step factor (`ms` ∈ [0, 0.002]).
- Semantics are computed once and cached.

### NeuralNetwork (`NeuralNetwork/NeuralNetwork.py`)

The base network for initial population individuals. It:
- Has a configurable number of hidden layers and neurons.
- Exposes `hidden_outputs` (intermediate layer activations) used as input to the first Block.
- Is trained via supervised learning before entering the evolutionary loop.

### Mutation Operators (`individual/mutation_operators.py`)

- **`inflate_mutation`**: Creates a new random `Block`, trains it, computes its semantics, and concatenates to the individual's total semantics.
- **`deflate_mutation`**: Randomly selects and removes a block; slices corresponding columns from semantics. For multiclass, removes `n_classes` columns at a time.

### Multiclass Semantic Normalization

For multiclass problems, each block's semantics are softmax-normalized. The first block uses standard softmax; subsequent blocks use centered softmax (output − 1/n_classes). Total semantics are summed across blocks for final prediction via cross-entropy loss.

### Evaluators (`utils/evaluators.py`)

- `binarized_rmse()` — RMSE with sigmoid binarization (binary).
- `binarized_bce()` — Binary cross-entropy with binarization (binary).
- `binarized_mcc()` — Matthews Correlation Coefficient (binary).
- `cross_entropy()` — Cross-entropy loss (multiclass).

### Data

Pre-computed ResNet features are stored as CSVs in `datasets/`:
- Datasets: `brain_tumor`, `tuberculosis`, `mc_tuberculosis`
- Feature extractors: `rn18`, `rn34`, `rn50`

Data is loaded, preprocessed with `StandardScaler`, and split into train/test before the evolutionary run. An additional 20% of train is reserved as a validation split for initial network training.

### Logging

Results are appended to CSV files in `log/`. Each row records per-generation metrics: `fitness`, `test_fitness`, `size`, `length`, `hidden_layer_count`. Parallel scripts use `filelock` for thread-safe writes and a checkpoint mechanism to skip already-completed configurations.

### Noise Robustness (`noise_utils.py`)

Utilities for injecting Gaussian and Poisson noise into features. Used by `scripts/baselines/noise_baseline.py` and `scripts/dslm/noise_main.py` to evaluate robustness.

## Key Configuration Parameters (set in run scripts)

| Parameter | Description |
|-----------|-------------|
| `pop_size` | Population size |
| `n_iter` | Number of generations |
| `ms` | Mutation step factor (Block output scale) |
| `n_elites` | Number of elite individuals preserved |
| `inflate_prob` | Probability of inflate vs. deflate mutation |
| `pool_size` | Tournament selection pool size |
| `n_jobs` | Parallelism for initial population training (joblib) |

## Dependencies

See `requirements.txt`. Key libraries:
- PyTorch 2.7.0, torchvision 0.22.0
- scikit-learn 1.6.1
- numpy 2.2.5, pandas 2.2.3
- joblib 1.5.0 (population parallelization)
- filelock (parallel logging)
