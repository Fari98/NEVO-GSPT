import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split as sklearn_split
from utils.evaluators import binarized_mcc, binarized_rmse, binarized_bce
from utils.info import logger
import datetime
import time
import torch.optim as optim

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

# Dataset configuration
sample_sizes = [10, 50, 100, 250, 500]
seeds = 30


class BrainTumorImageLoader:
    """Load brain tumor images."""

    def __init__(self, path):
        self.path = Path(path)

    def get_images_and_labels(self):
        """Return list of (image_path, label) tuples."""
        images = []

        for label_dir in self.path.iterdir():
            if label_dir.is_dir() and label_dir.name in ['yes', 'no']:
                label = 1 if label_dir.name == 'yes' else 0

                for img_file in label_dir.rglob('*.jpg'):
                    images.append((str(img_file), label))
                for img_file in label_dir.rglob('*.jpeg'):
                    images.append((str(img_file), label))
                for img_file in label_dir.rglob('*.png'):
                    images.append((str(img_file), label))

        return images


class TuberculosisImageLoader:
    """Load tuberculosis images."""

    def __init__(self, path):
        self.path = Path(path)
        self.metadata_path = self.path / 'shenzhen_metadata.csv'
        nested_images_dir = self.path / 'images' / 'images'
        self.images_dir = nested_images_dir if nested_images_dir.exists() else self.path / 'images'

    def get_images_and_labels(self):
        """Return list of (image_path, label) tuples."""
        images = []

        if self.metadata_path.exists():
            try:
                df = pd.read_csv(self.metadata_path)

                for idx, row in df.iterrows():
                    img_name = row['study_id']
                    findings = row['findings']
                    label = 0 if str(findings).lower().strip() == 'normal' else 1

                    if img_name:
                        img_path = self.images_dir / f"{img_name}.png"
                        if not img_path.exists():
                            img_path = self.images_dir / str(img_name)

                        if img_path.exists():
                            images.append((str(img_path), label))

            except Exception as e:
                print(f"Error reading metadata: {e}")

        return images


class ImageDataset(Dataset):
    """PyTorch Dataset for loading images."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.FloatTensor([label])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), torch.FloatTensor([label])
            return Image.new('RGB', (224, 224)), torch.FloatTensor([label])


class ViTBinaryClassifier(nn.Module):
    """Vision Transformer for binary classification"""

    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pretrained ViT model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Replace classification head for binary classification
        # Original head outputs 1000 classes, we need 1 output with sigmoid
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            # Unfreeze the new head
            for param in self.vit.heads.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.vit(x)


def _run(seed, dataset_name, sample_size, dataset_path=None, freeze_backbone=True):
    """
    Run ViT baseline experiment

    Args:
        seed: Random seed
        dataset_name: 'brain_tumor' or 'tuberculosis'
        sample_size: Number of training samples
        dataset_path: Path to dataset (if None, will download)
        freeze_backbone: Whether to freeze ViT backbone (only train head)
    """

    dataset_full = f"{dataset_name}_vit_{sample_size}"

    # Download and load dataset images (if path not provided)
    if dataset_path is None:
        if dataset_name == 'brain_tumor':
            dataset_path = kagglehub.dataset_download("ahmedhamada0/brain-tumor-detection")
        elif dataset_name == 'tuberculosis':
            dataset_path = kagglehub.dataset_download("raddar/tuberculosis-chest-xrays-shenzhen")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load images
    if dataset_name == 'brain_tumor':
        image_loader = BrainTumorImageLoader(dataset_path)
    elif dataset_name == 'tuberculosis':
        image_loader = TuberculosisImageLoader(dataset_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get all image paths and labels
    images_labels = image_loader.get_images_and_labels()
    image_paths = [img for img, _ in images_labels]
    labels = np.array([lbl for _, lbl in images_labels])

    print(f"Loaded {len(image_paths)} images for {dataset_name}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Split data: train/test split based on sample_size
    # Keep sample_size images for training, rest for testing
    indices = np.arange(len(image_paths))
    train_indices, test_indices = sklearn_split(
        indices,
        train_size=sample_size,
        random_state=seed,
        stratify=labels
    )

    # Further split training into train/val (80/20)
    train_labels = labels[train_indices]
    train_indices_final, val_indices = sklearn_split(
        train_indices,
        test_size=0.2,
        random_state=seed,
        stratify=train_labels
    )

    # Get paths and labels for each split
    train_paths = [image_paths[i] for i in train_indices_final]
    train_labels = labels[train_indices_final]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = labels[val_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = labels[test_indices]

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # Define transforms for ViT (same as ImageNet preprocessing)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels, transform=transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform=transform)
    test_dataset = ImageDataset(test_paths, test_labels, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTBinaryClassifier(freeze_backbone=freeze_backbone).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()

    if freeze_backbone:
        # Only optimize head parameters
        optimizer = optim.AdamW(model.vit.heads.parameters(), lr=1e-3)
    else:
        # Optimize all parameters
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training
    num_epochs = 1000
    history = {'train_loss': [], 'val_loss': [], 'test_loss': []}

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        # Test phase
        test_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        epoch_test_loss = test_loss / len(test_loader)
        history['test_loss'].append(epoch_test_loss)

        # Log evolution
        logger(f'log/baseline_vit_evo_{day}.csv',
               generation=epoch,
               timing=0,
               run_info=[dataset_full, epoch_train_loss, epoch_val_loss, epoch_test_loss],
               seed=seed)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, "
                  f"Test Loss: {epoch_test_loss:.4f}")

    end_time = time.time()

    # Final evaluation
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    y_pred = torch.cat(all_preds).flatten().numpy()
    y_test_np = torch.cat(all_targets).numpy()

    # Calculate metrics
    rmse = binarized_rmse()(y_test_np, y_pred)
    bce = binarized_bce()(y_test_np, y_pred)
    mcc = binarized_mcc()(y_test_np, y_pred)

    # Log final results
    logger(f'log/baseline_vit_{day}.csv',
           generation=1000,
           timing=end_time - start_time,
           run_info=[dataset_full, rmse, bce, mcc],
           seed=seed)

    print(f"Completed {dataset_full} - Seed {seed}")
    print(f"RMSE: {rmse:.4f}, BCE: {bce:.4f}, MCC: {mcc:.4f}")

    return 'done'