import torch
import torch.nn as nn
import torch.optim as optim
from datasets.data_loader_resnet import *
from utils.utils import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import datetime
import pandas as pd
import os

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

loaders = [load_brain_tumor, load_tuberculosis]
resnet_versions = ['rn18', 'rn34', 'rn50']
sizes = [10, 50, 100, 250, 500]
seeds = 5


class SimpleClassifier(nn.Module):
    """Simple linear classifier for few-shot learning"""
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def euclidean_distance(x, y):
    """Compute euclidean distance between two tensors"""
    return torch.cdist(x, y, p=2)


def protonet(X_train, y_train, X_test, y_test, n_classes):
    """
    Prototypical Networks: Classify based on distance to class prototypes (mean embeddings)
    """
    # Compute class prototypes (mean of each class)
    prototypes = []
    for c in range(n_classes):
        mask = y_train == c
        if mask.sum() > 0:
            prototype = X_train[mask].mean(dim=0)
            prototypes.append(prototype)
        else:
            # If class not in training, use zero vector
            prototypes.append(torch.zeros(X_train.shape[1]))

    prototypes = torch.stack(prototypes)  # [n_classes, feature_dim]

    # Compute distances and predictions for train set
    distances_train = euclidean_distance(X_train, prototypes)
    predictions_train = torch.argmin(distances_train, dim=1)

    # Compute distances and predictions for test set
    distances_test = euclidean_distance(X_test, prototypes)
    predictions_test = torch.argmin(distances_test, dim=1)

    return predictions_train, predictions_test


def matching_networks(X_train, y_train, X_test, y_test, n_classes):
    """
    Matching Networks: Classify based on attention-weighted nearest neighbors
    Simple version using cosine similarity
    """
    # Normalize features for cosine similarity
    X_train_norm = torch.nn.functional.normalize(X_train, dim=1)
    X_test_norm = torch.nn.functional.normalize(X_test, dim=1)

    # Predictions for train set
    similarities_train = torch.mm(X_train_norm, X_train_norm.t())
    attention_train = torch.nn.functional.softmax(similarities_train, dim=1)
    predictions_train = []
    for i in range(X_train.shape[0]):
        votes = torch.zeros(n_classes)
        for j in range(X_train.shape[0]):
            votes[y_train[j].long()] += attention_train[i, j]
        predictions_train.append(torch.argmax(votes))
    predictions_train = torch.tensor(predictions_train)

    # Predictions for test set
    similarities_test = torch.mm(X_test_norm, X_train_norm.t())
    attention_test = torch.nn.functional.softmax(similarities_test, dim=1)
    predictions_test = []
    for i in range(X_test.shape[0]):
        votes = torch.zeros(n_classes)
        for j in range(X_train.shape[0]):
            votes[y_train[j].long()] += attention_test[i, j]
        predictions_test.append(torch.argmax(votes))
    predictions_test = torch.tensor(predictions_test)

    return predictions_train, predictions_test


def maml(X_train, y_train, X_test, y_test, n_classes, inner_lr=0.01, outer_steps=20):
    """
    MAML (Model-Agnostic Meta-Learning):
    Simple version - fine-tune a model on support set, evaluate on query set
    """
    input_dim = X_train.shape[1]
    model = SimpleClassifier(input_dim, n_classes)

    # Meta-training (simplified - just multiple gradient steps)
    optimizer = optim.SGD(model.parameters(), lr=inner_lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(outer_steps):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.long())
        loss.backward()
        optimizer.step()

    # Evaluate on both train and test sets
    model.eval()
    with torch.no_grad():
        outputs_train = model(X_train)
        predictions_train = torch.argmax(outputs_train, dim=1)

        outputs_test = model(X_test)
        predictions_test = torch.argmax(outputs_test, dim=1)

    return predictions_train, predictions_test


def transductive_finetune(X_train, y_train, X_test, y_test, n_classes, epochs=100, lr=0.001):
    """
    Transductive Fine-Tuning: Train classifier on support set, evaluate on query set
    """
    input_dim = X_train.shape[1]
    model = SimpleClassifier(input_dim, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train on support set (X_train, y_train)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.long())
        loss.backward()
        optimizer.step()

    # Evaluate on both train and test sets
    model.eval()
    with torch.no_grad():
        outputs_train = model(X_train)
        predictions_train = torch.argmax(outputs_train, dim=1)

        outputs_test = model(X_test)
        predictions_test = torch.argmax(outputs_test, dim=1)

    return predictions_train, predictions_test


def log_results(path, row):
    """Log results to CSV file"""
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, mode='w', header=True, index=False)


def _run(seed, loader, resnet_v, size):
    """Run all few-shot baselines on a dataset"""

    # Load data
    X, y = loader(model_type=resnet_v, X_y=True)
    dataset = loader.__name__.split("load_")[-1] + '_' + resnet_v + '_' + str(size)

    # Standardize features
    X = StandardScaler().fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=size, random_state=seed, stratify=y
    )

    # Convert to tensors
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).squeeze()
    y_test = torch.Tensor(y_test).squeeze()

    n_classes = len(torch.unique(y))

    print(f"\nRunning: {dataset} - seed={seed}")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}, Classes: {n_classes}")

    results = {
        'dataset': dataset,
        'seed': seed,
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'n_classes': n_classes
    }

    # ProtoNet
    print("  Running ProtoNet...")
    pred_protonet_train, pred_protonet_test = protonet(X_train, y_train, X_test, y_test, n_classes)
    mcc_protonet_train = matthews_corrcoef(y_train.numpy(), pred_protonet_train.numpy())
    mcc_protonet_test = matthews_corrcoef(y_test.numpy(), pred_protonet_test.numpy())
    results['protonet_train_mcc'] = mcc_protonet_train
    results['protonet_test_mcc'] = mcc_protonet_test
    print(f"    ProtoNet - Train MCC: {mcc_protonet_train:.4f}, Test MCC: {mcc_protonet_test:.4f}")

    # Matching Networks
    # print("  Running Matching Networks...")
    # pred_matching_train, pred_matching_test = matching_networks(X_train, y_train, X_test, y_test, n_classes)
    # mcc_matching_train = matthews_corrcoef(y_train.numpy(), pred_matching_train.numpy())
    # mcc_matching_test = matthews_corrcoef(y_test.numpy(), pred_matching_test.numpy())
    # results['matching_train_mcc'] = mcc_matching_train
    # results['matching_test_mcc'] = mcc_matching_test
    # print(f"    Matching Networks - Train MCC: {mcc_matching_train:.4f}, Test MCC: {mcc_matching_test:.4f}")

    # MAML
    # print("  Running MAML...")
    # pred_maml_train, pred_maml_test = maml(X_train, y_train, X_test, y_test, n_classes, inner_lr=0.01, outer_steps=20)
    # mcc_maml_train = matthews_corrcoef(y_train.numpy(), pred_maml_train.numpy())
    # mcc_maml_test = matthews_corrcoef(y_test.numpy(), pred_maml_test.numpy())
    # results['maml_train_mcc'] = mcc_maml_train
    # results['maml_test_mcc'] = mcc_maml_test
    # print(f"    MAML - Train MCC: {mcc_maml_train:.4f}, Test MCC: {mcc_maml_test:.4f}")
    #
    # # Transductive Fine-Tuning
    # print("  Running Transductive Fine-Tuning...")
    # pred_transductive_train, pred_transductive_test = transductive_finetune(X_train, y_train, X_test, y_test, n_classes, epochs=100, lr=0.001)
    # mcc_transductive_train = matthews_corrcoef(y_train.numpy(), pred_transductive_train.numpy())
    # mcc_transductive_test = matthews_corrcoef(y_test.numpy(), pred_transductive_test.numpy())
    # results['transductive_train_mcc'] = mcc_transductive_train
    # results['transductive_test_mcc'] = mcc_transductive_test
    # print(f"    Transductive Fine-Tuning - Train MCC: {mcc_transductive_train:.4f}, Test MCC: {mcc_transductive_test:.4f}")

    # Log results
    log_path = f'log/{day}_fs_baseline.csv'
    log_results(log_path, results)

    print(f"Completed: {dataset} - seed={seed}\n")
    return results


# if __name__ == '__main__':
#     # Run experiments
#     for loader in [loaders[0]]:
#         for resnet_v in [resnet_versions[0]]:
#             for size in [sizes[0]]:
#                 for seed in range(1):
#                     _run(seed, loader, resnet_v, size)
#
#     print("All experiments completed!")