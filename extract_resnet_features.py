"""
Extract CNN features from ResNet18 for brain tumor and tuberculosis datasets.
Features are saved along with targets for DSLM training.
"""

import os
import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import kagglehub
from pathlib import Path
import pickle
import pandas as pd


class ResNetFeatureExtractor:
    """Extract CNN features from ResNet models (without fully connected layer)."""

    def __init__(self, model_name='resnet18', device='cpu'):
        self.device = device
        self.model_name = model_name

        # Load the appropriate model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Remove the fully connected layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path):
        """Extract features from a single image."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(img_tensor)

            return features.squeeze(0).cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


class BrainTumorDataset:
    """Handle brain tumor detection dataset."""

    def __init__(self, path):
        self.path = Path(path)

    def get_images_and_labels(self):
        """Return list of (image_path, label) tuples."""
        images = []

        # Check for yes/no directories at root level
        for label_dir in self.path.iterdir():
            if label_dir.is_dir() and label_dir.name in ['yes', 'no']:
                label = 1 if label_dir.name == 'yes' else 0

                # Search for images recursively
                for img_file in label_dir.rglob('*.jpg'):
                    images.append((str(img_file), label))
                for img_file in label_dir.rglob('*.jpeg'):
                    images.append((str(img_file), label))
                for img_file in label_dir.rglob('*.png'):
                    images.append((str(img_file), label))

        return images


class TuberculosisDataset:
    """Handle tuberculosis chest X-rays dataset."""

    def __init__(self, path):
        self.path = Path(path)
        self.metadata_path = self.path / 'shenzhen_metadata.csv'
        # Try images/images first (nested structure), then images
        nested_images_dir = self.path / 'images' / 'images'
        self.images_dir = nested_images_dir if nested_images_dir.exists() else self.path / 'images'

    def get_images_and_labels(self):
        """Return list of (image_path, label) tuples using metadata CSV."""
        images = []

        # Try to read metadata CSV
        if self.metadata_path.exists():
            try:
                df = pd.read_csv(self.metadata_path)
                print(f"  Metadata columns: {df.columns.tolist()}")
                print(f"  Metadata shape: {df.shape}")

                # The first column is study_id (filename), findings column contains TB labels
                for idx, row in df.iterrows():
                    img_name = row['study_id']  # study_id is the filename

                    # Get label from findings column
                    # "normal" = 0 (no TB), anything else = 1 (TB)
                    findings = row['findings']
                    label = 0 if str(findings).lower().strip() == 'normal' else 1

                    if img_name:
                        # Try with .png extension
                        img_path = self.images_dir / f"{img_name}.png"
                        if not img_path.exists():
                            # Try without extension if filename already has it
                            img_path = self.images_dir / str(img_name)

                        if img_path.exists():
                            images.append((str(img_path), label))

            except Exception as e:
                print(f"  Error reading metadata CSV: {e}")

        return images


class TuberculosisMulticlassDataset:
    """Handle tuberculosis chest X-rays dataset with multiclass labels (4 classes)."""

    def __init__(self, path):
        self.path = Path(path)
        self.metadata_path = self.path / 'shenzhen_metadata.csv'
        # Try images/images first (nested structure), then images
        nested_images_dir = self.path / 'images' / 'images'
        self.images_dir = nested_images_dir if nested_images_dir.exists() else self.path / 'images'

    @staticmethod
    def classify_finding(finding):
        """Classify finding into one of 4 classes:
        0: Normal
        1: Localized TB (right/left/upper/lower)
        2: Bilateral/Disseminated TB
        3: With complications (cavitation, effusion, pneumothorax, etc.)
        """
        finding_lower = str(finding).lower().strip()

        # Class 0: Normal
        if finding_lower == 'normal':
            return 0

        # Class 3: With complications - check for specific keywords
        complications = ['cavity', 'cavitation', 'effusion', 'pneumothorax',
                        'pleural thickening', 'pleurisy', 'adhesions', 'atelectasis',
                        'encapsulated', 'fibrous changes', 'hematogenous', 'disseminated',
                        'hyperplastic', 'copd', 'pneumonia', 'emphysema']

        has_complications = any(comp in finding_lower for comp in complications)

        # Class 2: Bilateral or Disseminated
        is_bilateral = 'bilateral' in finding_lower
        is_disseminated = any(word in finding_lower for word in ['disseminated', 'hematogenous', 'widespread'])

        if is_bilateral or is_disseminated:
            if has_complications:
                return 3  # Bilateral/Disseminated with complications
            else:
                return 2  # Bilateral/Disseminated without complications

        # Class 1: Localized TB (right, left, upper, lower fields)
        if has_complications:
            return 3  # Localized with complications
        else:
            return 1  # Localized TB without complications

    def get_images_and_labels(self):
        """Return list of (image_path, label) tuples using metadata CSV."""
        images = []

        # Try to read metadata CSV
        if self.metadata_path.exists():
            try:
                df = pd.read_csv(self.metadata_path)
                print(f"  Metadata columns: {df.columns.tolist()}")
                print(f"  Metadata shape: {df.shape}")

                # Track label distribution
                label_counts = {0: 0, 1: 0, 2: 0, 3: 0}

                # The first column is study_id (filename), findings column contains TB labels
                for idx, row in df.iterrows():
                    img_name = row['study_id']  # study_id is the filename
                    findings = row['findings']

                    # Classify the finding into one of 4 classes
                    label = self.classify_finding(findings)
                    label_counts[label] += 1

                    if img_name:
                        # Try with .png extension
                        img_path = self.images_dir / f"{img_name}.png"
                        if not img_path.exists():
                            # Try without extension if filename already has it
                            img_path = self.images_dir / str(img_name)

                        if img_path.exists():
                            images.append((str(img_path), label))

                # Print the class distribution
                class_names = {0: 'Normal', 1: 'Localized TB', 2: 'Bilateral/Disseminated TB', 3: 'With Complications'}
                print(f"  Class distribution:")
                for cls in range(4):
                    print(f"    {cls} ({class_names[cls]}): {label_counts[cls]} images")

            except Exception as e:
                print(f"  Error reading metadata CSV: {e}")

        return images


def extract_and_save_features(dataset_class, dataset_path, output_name, model_name='resnet18', device='cpu'):
    """
    Extract features from all images and save as CSV file.
    CSV format: [feature_0, feature_1, ..., feature_N, target]

    Args:
        dataset_class: Dataset class (BrainTumorDataset or TuberculosisDataset)
        dataset_path: Path to downloaded dataset
        output_name: Name for output files (without extension)
        model_name: ResNet model to use ('resnet18', 'resnet34', or 'resnet50')
        device: 'cpu' or 'cuda'
    """

    print(f"\n{'='*60}")
    print(f"Processing {output_name} dataset with {model_name}...")
    print(f"{'='*60}")

    # Initialize
    dataset = dataset_class(dataset_path)
    images_labels = dataset.get_images_and_labels()

    if not images_labels:
        print(f"No images found in {dataset_path}")
        return

    print(f"Found {len(images_labels)} images")

    extractor = ResNetFeatureExtractor(model_name=model_name, device=device)

    features_list = []
    targets_list = []
    failed_count = 0

    for idx, (img_path, label) in enumerate(images_labels):
        if (idx + 1) % 50 == 0:
            print(f"Processing image {idx + 1}/{len(images_labels)}...")

        features = extractor.extract_features(img_path)

        if features is not None:
            features_list.append(features)
            targets_list.append(label)
        else:
            failed_count += 1

    # Convert to numpy arrays
    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets_list, dtype=np.int32)

    print(f"\nSuccessfully processed: {len(X)} images")
    print(f"Failed: {failed_count} images")
    print(f"Feature shape per image: {X[0].shape}")
    print(f"Unique labels: {np.unique(y)}")
    print(f"Label distribution: {np.bincount(y)}")

    # Create CSV with features and target
    output_dir = Path('datasets/resnet_features')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create column names: feature_0, feature_1, ..., feature_N, target
    n_features = X.shape[1]
    column_names = [f'feature_{i}' for i in range(n_features)] + ['target']

    # Combine features and targets
    data = np.column_stack([X, y])

    # Map model names to file suffixes
    model_suffix = {'resnet18': 'rn18', 'resnet34': 'rn34', 'resnet50': 'rn50'}
    suffix = model_suffix.get(model_name, model_name)

    # Save as CSV
    csv_path = output_dir / f'{output_name}_{suffix}.csv'
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(csv_path, index=False)

    print(f"\nSaved CSV to: {csv_path}")
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {column_names[:5]} ... {column_names[-1]}")

    return X, y


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Models to extract features from
    models_to_use = ['resnet18', 'resnet34', 'resnet50']

    # Download and process brain tumor dataset
    # print("\nDownloading brain tumor dataset...")
    # brain_tumor_path = kagglehub.dataset_download("ahmedhamada0/brain-tumor-detection")
    # for model in models_to_use:
    #     extract_and_save_features(BrainTumorDataset, brain_tumor_path, 'brain_tumor', model_name=model, device=device)

    # Download and process tuberculosis dataset
    print("\nDownloading tuberculosis dataset...")
    tuberculosis_path = kagglehub.dataset_download("raddar/tuberculosis-chest-xrays-shenzhen")

    # # Binary classification (normal vs TB)
    # for model in models_to_use:
    #     extract_and_save_features(TuberculosisDataset, tuberculosis_path, 'tuberculosis', model_name=model, device=device)

    # Multiclass classification (normal + all TB types)
    for model in models_to_use:
        extract_and_save_features(TuberculosisMulticlassDataset, tuberculosis_path, 'mc_tuberculosis', model_name=model, device=device)

    print("\n" + "="*60)
    print("Feature extraction complete!")
    print("Data saved to datasets/resnet_features/")
    print("Generated files:")
    print("  - brain_tumor_rn18.csv, brain_tumor_rn34.csv, brain_tumor_rn50.csv")
    print("  - tuberculosis_rn18.csv, tuberculosis_rn34.csv, tuberculosis_rn50.csv")
    print("  - mc_tuberculosis_rn18.csv, mc_tuberculosis_rn34.csv, mc_tuberculosis_rn50.csv")
    print("="*60)


if __name__ == "__main__":
    main()