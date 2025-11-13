"""
Runtime Comparison Script: DSLM vs NN with Backpropagation
Compares:
1. Overall algorithm runtime (100 generations/epochs)
2. Individual evaluation runtime (inflate/deflate mutation vs backpropagation)
"""

import torch
import time
import numpy as np
from DSLM import DSLM
from datasets.data_loader import load_airfoil
from utils.utils import StandardScaler, train_test_split, uniform_random_step_generator, rmse
from population.initializers import initialize_population
from population.selection_algorithms import torunament_selection
from individual.mutation_operators import deflate_mutation, inflate_mutation
from NeuralNetwork.utils import create_network
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from torch import nn, optim
import datetime

# Setup
now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

NUM_RUNS = 30
NUM_GENERATIONS = 100
NUM_EPOCHS = 100
DATASET = load_airfoil

results = {
    'dslm_total_times': [],
    'nn_total_times': [],
    'dslm_eval_times': [],
    'nn_eval_times': [],
}

print("=" * 80)
print("Runtime Comparison: DSLM vs NN with Backpropagation")
print("=" * 80)
print(f"Dataset: Airfoil")
print(f"Number of runs: {NUM_RUNS}")
print(f"DSLM generations: {NUM_GENERATIONS}")
print(f"NN epochs: {NUM_EPOCHS}")
print("=" * 80)

for run in range(NUM_RUNS):
    print(f"\n{'='*80}")
    print(f"RUN {run + 1}/{NUM_RUNS}")
    print(f"{'='*80}")

    # Load and prepare data
    X, y = DATASET(X_y=True)
    dataset_name = DATASET.__name__.split("load_")[-1]
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed=run)
    X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, p_test=0.2, seed=run)

    # ============================================================================
    # DSLM BENCHMARK
    # ============================================================================
    print("\n[DSLM] Initializing optimizer...")

    # Time the initialization
    init_start = time.time()

    optimizer = DSLM(
        initializer=initialize_population(
            X_train_nn,
            y_train_nn,
            maximum_width=16,
            maximum_depth=3,
            activation_functions=[nn.ReLU()],
            pretrain_part=0,
            X_val=X_val, y_val=y_val,
            epochs=10, batch_size=32, learning_rate=0.001,
            criterion=nn.MSELoss(),
            optimizer=optim.Adam,
            device='cpu'
        ),
        selector=torunament_selection(2),
        inflate_mutator=inflate_mutation(X_train, ms_generator=uniform_random_step_generator(0, 2), X_test=X_test),
        deflate_mutator=deflate_mutation,
        crossover=None,
        p_m=1,
        p_im=0.5,
        p_dm=0.5,
        p_xo=0,
        pop_size=100,
        seed=run,
    )

    init_end = time.time()
    print(f"[DSLM] Initialization time: {init_end - init_start:.4f}s")

    # Time the main algorithm
    print(f"[DSLM] Running optimization for {NUM_GENERATIONS} generations...")
    dslm_start = time.time()

    optimizer.solve(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metric=rmse,
        max_depth=None,
        generations=NUM_GENERATIONS,
        elitism=True,
        dataset_name=dataset_name,
        log=0,  # Disable logging for cleaner output
        verbose=0,
        n_jobs=1  # Use single job to avoid parallelization overhead
    )

    dslm_end = time.time()
    dslm_total_time = dslm_end - dslm_start
    results['dslm_total_times'].append(dslm_total_time)

    print(f"[DSLM] Total runtime (100 gen): {dslm_total_time:.4f}s")

    # Measure individual evaluation time for DSLM (inflate/deflate mutations)
    print("\n[DSLM] Measuring individual evaluation time...")

    # Get the inflate and deflate mutators
    inflate_mut = optimizer.inflate_mutator
    deflate_mut = optimizer.deflate_mutator

    # Use an individual from the population
    test_individual = optimizer.population.individuals[0]

    # Measure inflate mutation
    inflate_times = []
    for _ in range(30):
        eval_start = time.time()
        _ = inflate_mut(test_individual)
        eval_end = time.time()
        inflate_times.append(eval_end - eval_start)

    avg_inflate_time = np.median(inflate_times)

    # Measure deflate mutation
    deflate_times = []
    for _ in range(30):
        eval_start = time.time()
        _ = deflate_mut(test_individual)
        eval_end = time.time()
        deflate_times.append(eval_end - eval_start)

    avg_deflate_time = np.median(deflate_times)
    avg_dslm_eval_time = (avg_inflate_time + avg_deflate_time) / 2
    results['dslm_eval_times'].append(avg_dslm_eval_time)

    print(f"[DSLM] Avg inflate mutation time: {avg_inflate_time:.6f}s")
    print(f"[DSLM] Avg deflate mutation time: {avg_deflate_time:.6f}s")
    print(f"[DSLM] Avg evaluation time: {avg_dslm_eval_time:.6f}s")

    # ============================================================================
    # NEURAL NETWORK BENCHMARK
    # ============================================================================
    print("\n[NN] Creating neural network...")

    # Create a network with similar architecture
    nn_model = NeuralNetwork(create_network(X_train.shape[1], width=16, depth=3))

    print(f"[NN] Training network for {NUM_EPOCHS} epochs...")
    nn_start = time.time()

    nn_model.train_network(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        epochs=NUM_EPOCHS,
        batch_size=32,
        learning_rate=0.001,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam,
        device='cpu'
    )

    nn_end = time.time()
    nn_total_time = nn_end - nn_start
    results['nn_total_times'].append(nn_total_time)

    print(f"[NN] Total runtime (100 epochs): {nn_total_time:.4f}s")

    # Measure individual evaluation time for NN (backpropagation on one batch)
    print("\n[NN] Measuring individual evaluation time (backpropagation)...")

    # Use a small batch for measurement
    X_batch = torch.tensor(X_train, dtype=torch.float32)
    y_batch = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    nn_model.eval()  # Set to eval mode first
    criterion = nn.MSELoss()

    backprop_times = []
    for _ in range(10):
        eval_start = time.time()
        nn_model.train()  # Set to training mode

        output = nn_model.forward(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()

        eval_end = time.time()
        backprop_times.append(eval_end - eval_start)

    avg_nn_eval_time = np.median(backprop_times)
    results['nn_eval_times'].append(avg_nn_eval_time)

    print(f"[NN] Avg backpropagation time (batch of 32): {avg_nn_eval_time:.6f}s")

    # ============================================================================
    # PRINT RUN SUMMARY
    # ============================================================================
    print(f"\n{'-'*80}")
    print(f"RUN {run + 1} SUMMARY:")
    print(f"{'-'*80}")
    print(f"DSLM Total Time:        {dslm_total_time:>12.4f}s")
    print(f"NN Total Time:          {nn_total_time:>12.4f}s")
    print(f"Speedup (NN/DSLM):      {dslm_total_time/nn_total_time:>12.2f}x")
    print(f"\nDSLM Eval Time:         {avg_dslm_eval_time:>12.6f}s")
    print(f"NN Eval Time:           {avg_nn_eval_time:>12.6f}s")
    print(f"Eval Speedup (NN/DSLM): {avg_dslm_eval_time/avg_nn_eval_time:>12.2f}x")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n\n{'='*80}")
print("FINAL RESULTS (Averaged across all runs)")
print(f"{'='*80}\n")

avg_dslm_total = np.median(results['dslm_total_times'])
avg_nn_total = np.median(results['nn_total_times'])
avg_dslm_eval = np.median(results['dslm_eval_times'])
avg_nn_eval = np.median(results['nn_eval_times'])

std_dslm_total = np.std(results['dslm_total_times'])
std_nn_total = np.std(results['nn_total_times'])
std_dslm_eval = np.std(results['dslm_eval_times'])
std_nn_eval = np.std(results['nn_eval_times'])

print("OVERALL ALGORITHM RUNTIME (100 gen/epochs):")
print(f"  DSLM:           {avg_dslm_total:.4f}s ± {std_dslm_total:.4f}s")
print(f"  NN (backprop):  {avg_nn_total:.4f}s ± {std_nn_total:.4f}s")
print(f"  Speedup:        {avg_dslm_total/avg_nn_total:.2f}x (NN is {avg_dslm_total/avg_nn_total:.2f}x faster)")

print("\nINDIVIDUAL EVALUATION TIME:")
print(f"  DSLM (inflate/deflate): {avg_dslm_eval:.6f}s ± {std_dslm_eval:.6f}s")
print(f"  NN (backprop on batch): {avg_nn_eval:.6f}s ± {std_nn_eval:.6f}s")
print(f"  Speedup:                {avg_dslm_eval/avg_nn_eval:.2f}x (NN is {avg_dslm_eval/avg_nn_eval:.2f}x faster)")

print("\n" + "="*80)
print("Individual run times:")
print("="*80)
print("\nDSLM Total Times (seconds):")
for i, t in enumerate(results['dslm_total_times'], 1):
    print(f"  Run {i}: {t:.4f}s")

print("\nNN Total Times (seconds):")
for i, t in enumerate(results['nn_total_times'], 1):
    print(f"  Run {i}: {t:.4f}s")

print("\nDSLM Eval Times (seconds):")
for i, t in enumerate(results['dslm_eval_times'], 1):
    print(f"  Run {i}: {t:.6f}s")

print("\nNN Eval Times (seconds):")
for i, t in enumerate(results['nn_eval_times'], 1):
    print(f"  Run {i}: {t:.6f}s")

print("\n" + "="*80)
