"""
Comprehensive GAN Evaluation Script

This script calculates multiple evaluation metrics for GANs:
- FID (Fréchet Inception Distance): Distribution similarity
- IS (Inception Score): Quality and diversity  
- Precision: Fraction of realistic generated images
- Recall: Coverage of real data manifold

Usage:
    python scripts/fidscore.py --model path/to/generator.pth --config path/to/config.json --num-samples 1000
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.utils import save_image 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import re 
# --- 1. SETUP PATHS ---
project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from datetime import datetime

def get_timestamp_from_path(model_path):
    """
    Extracts a timestamp (YYYYMMDD_HHMMSS) from the model path if it exists.
    If not found, returns the current timestamp as a fallback.
    """
    print("Attempting to extract timestamp from model path...")
    path = Path(model_path)
    # Regex to find a pattern like 20251107_034414
    timestamp_pattern = re.compile(r'\d{8}_\d{6}')

    # Iterate through the parts of the path to find the timestamp
    for part in path.parts:
        match = timestamp_pattern.search(part)
        if match:
            extracted_timestamp = match.group(0)
            print(f"  Found timestamp in path: {extracted_timestamp}")
            return extracted_timestamp

    print("  No timestamp found in path. Using current time.")
    # Fallback to current time if no timestamp is found in the path
    return datetime.now().strftime("%Y%m%d_%H%M%S")



def save_comparison_grid(real_dataloader, fake_images, model_name, strategy_name, timestamp):
    """Save side-by-side comparison of real and fake images for debugging."""
    output_dir = Path("experiments/fid/fid_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect diverse real images (stratified sampling across batches)
    print("Collecting diverse real image samples...")
    real_samples = []
    target_samples = 80  # Show 80 images total
    
    # Try to get diverse samples by collecting from multiple batches
    for batch_idx, batch in enumerate(real_dataloader):
        real_samples.append(batch)
        if len(torch.cat(real_samples, dim=0)) >= target_samples:
            break
        if batch_idx >= 10:  # Don't iterate too many batches
            break
    
    real_images_diverse = torch.cat(real_samples, dim=0)[:target_samples]
    
    # Save real images grid
    real_grid_path = output_dir / f"real_samples_{model_name}_{strategy_name}_{timestamp}.png"
    save_image(real_images_diverse, real_grid_path, nrow=10, normalize=True)
    print(f"Real images grid saved to: {real_grid_path}")
    
    # Save fake images grid (random sample)
    fake_grid_path = output_dir / f"fake_samples_{model_name}_{strategy_name}_{timestamp}.png"
    # Randomly sample 80 fake images for comparison
    if len(fake_images) > target_samples:
        indices = torch.randperm(len(fake_images))[:target_samples]
        fake_samples_to_save = fake_images[indices]
    else:
        fake_samples_to_save = fake_images[:target_samples]
    
    save_image(fake_samples_to_save, fake_grid_path, nrow=10, normalize=True)
    print(f"Fake images grid saved to: {fake_grid_path}")
    
    # Print statistics for debugging
    print(f"\nImage Statistics:")
    print(f"   Real images - min: {real_images_diverse.min():.3f}, max: {real_images_diverse.max():.3f}, mean: {real_images_diverse.mean():.3f}")
    print(f"   Fake images - min: {fake_samples_to_save.min():.3f}, max: {fake_samples_to_save.max():.3f}, mean: {fake_samples_to_save.mean():.3f}")


def get_strategy_from_path(model_path):
    """Extracts the NAS strategy name from the model's parent directory path."""
    try:
        path = Path(model_path)
        strategy_name = path.parent.parent.parent.name
        return strategy_name
    except Exception:
        return "unknown" 


# --- 2. IMPORTS ---
try:
    from nepscript.models.factory import create_models_from_config
    from nepscript.utils.config import resolve_device
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# --- 3. DATASET CLASS WITH BETTER DIAGNOSTICS ---
class FIDDataset(Dataset):
    """Dataset for loading REAL images for FID calculation."""
    
    def __init__(self, data_root, csv_file, transform=None, max_samples=None, split=None):
        """
        Args:
            data_root: Root directory (e.g., 'data/dataset_1')
            csv_file: Path to CSV file with filename column
            transform: Image transformations
            max_samples: Limit number of samples
            split: Filter by split ('Train' or 'Test')
        """
        self.data_root = Path(data_root)
        self.transform = transform or transforms.ToTensor()
        
        # Load data from CSV
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded CSV with {len(df)} entries from {csv_file}")
            
            # Filter by split if specified
            if split is not None:
                df = df[df['filename'].str.contains(split, case=False)]
                print(f"   Filtered to {split} split: {len(df)} entries")
                
        except FileNotFoundError:
            print(f"Error: Labels CSV file not found at {csv_file}")
            self.image_paths = []
            self.labels = []
            return

        # Limit samples if specified
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"Randomly sampled {max_samples} images for FID calculation")

        # Construct full paths and validate
        self.image_paths = []
        self.labels = []
        missing_count = 0
        
        print(f"Validating image paths...")
        for idx, row in df.iterrows():
            fname = row['filename']
            full_path = self.data_root / fname
            
            if full_path.exists():
                self.image_paths.append(full_path)
                self.labels.append(row['label'])
            else:
                missing_count += 1
                if missing_count <= 3:  # Show first 3 missing files
                    print(f"Missing: {full_path}")
        
        if missing_count > 3:
            print(f"... and {missing_count - 3} more missing files")
        
        print(f"Found {len(self.image_paths)} valid images out of {len(df)} CSV entries")
        
        # Print label distribution
        if len(self.labels) > 0:
            label_counts = pd.Series(self.labels).value_counts().sort_index()
            print(f"Label distribution:")
            for label, count in label_counts.items():
                print(f"   Digit {label}: {count} images")
        
        if len(self.image_paths) == 0:
            print(f"\nCRITICAL ERROR: No valid images found!")
            print(f"   Data root: {self.data_root}")
            if len(df) > 0:
                print(f"   First CSV entry: {df['filename'].iloc[0]}")
                print(f"   Expected path: {self.data_root / df['filename'].iloc[0]}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a zero tensor matching expected shape
            return torch.zeros(1, 32, 32)


def get_inception_model(device, return_features=True):
    """Load pre-trained Inception-v3 for feature extraction or classification"""
    print("Loading Inception-v3 model...")
    model = inception_v3(weights='Inception_V3_Weights.DEFAULT', transform_input=False)
    
    if return_features:
        # For FID: remove final classifier, return 2048-dim features
        model.fc = nn.Identity()
        print("  Configured for feature extraction (FID, Precision/Recall)")
    else:
        # For IS: keep full classifier, return class predictions
        print("  Configured for classification (Inception Score)")
    
    model.eval()
    model = model.to(device)
    return model


def extract_features(images_or_dataloader, model, device, batch_size=50, is_real=True):
    """Extract features from images using Inception-v3"""
    model.eval()
    features = []
    
    image_type = "real" if is_real else "fake"
    
    if isinstance(images_or_dataloader, DataLoader):
        total_images = len(images_or_dataloader.dataset)
        data_iterator = images_or_dataloader
    else:
        total_images = len(images_or_dataloader)
        tensor_dataset = torch.utils.data.TensorDataset(images_or_dataloader)
        data_iterator = DataLoader(tensor_dataset, batch_size=batch_size)
    
    print(f"Extracting features from {total_images} {image_type} images...")

    processed_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iterator):
            if isinstance(batch, (list, tuple)):
                batch = batch[0].to(device)
            else:
                batch = batch.to(device)
            
            # Debug: Print stats for first batch
            if batch_idx == 0:
                print(f"   First batch shape: {batch.shape}")
                print(f"   First batch range: [{batch.min():.3f}, {batch.max():.3f}]")
            
            # Resize to 299x299 (Inception input size)
            batch_resized = torch.nn.functional.interpolate(
                batch, size=(299, 299), mode='bilinear', align_corners=False
            )
            
            # Convert grayscale to RGB
            if batch_resized.shape[1] == 1:
                batch_resized = batch_resized.repeat(1, 3, 1, 1)
            
            batch_features = model(batch_resized)
            features.append(batch_features.cpu().numpy())
            
            processed_count += len(batch)
            if (processed_count // batch_size) % 20 == 0 and processed_count > 0:
                print(f"  Processed {processed_count}/{total_images} images...")
    
    print(f"  Processed {processed_count}/{total_images} images... Done.")
    features_array = np.concatenate(features, axis=0)
    print(f"  Feature array shape: {features_array.shape}")
    
    return features_array


def calculate_statistics(features):
    """Calculate mean and covariance of features"""
    print("Calculating distribution statistics...")
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    print(f"  Mean shape: {mu.shape}, Covariance shape: {sigma.shape}")
    return mu, sigma


def calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6):
    """Calculate FID score between real and fake distributions"""
    print("Calculating FID score...")
    
    diff = mu_real - mu_fake
    
    covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if not np.isfinite(covmean).all():
        print("  Warning: covmean contains non-finite values, adding offset")
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = sqrtm((sigma_real + offset).dot(sigma_fake + offset))

    if np.iscomplexobj(covmean):
        print("  Warning: covmean is complex, taking real part")
        covmean = covmean.real
    
    trace_term = np.trace(sigma_real + sigma_fake - 2 * covmean)
    fid = diff.dot(diff) + trace_term
    
    print(f"  FID components: ||mu_diff||^2 = {diff.dot(diff):.2f}, trace = {trace_term:.2f}")
    
    return float(fid)


def extract_predictions(images_or_dataloader, model, device, batch_size=50):
    """Extract class predictions from images using Inception-v3 for IS calculation"""
    model.eval()
    predictions = []
    
    if isinstance(images_or_dataloader, DataLoader):
        total_images = len(images_or_dataloader.dataset)
        data_iterator = images_or_dataloader
    else:
        total_images = len(images_or_dataloader)
        tensor_dataset = torch.utils.data.TensorDataset(images_or_dataloader)
        data_iterator = DataLoader(tensor_dataset, batch_size=batch_size)
    
    print(f"Extracting predictions from {total_images} images for IS calculation...")

    processed_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iterator):
            if isinstance(batch, (list, tuple)):
                batch = batch[0].to(device)
            else:
                batch = batch.to(device)
            
            # Resize to 299x299 (Inception input size)
            batch_resized = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Convert grayscale to RGB
            if batch_resized.shape[1] == 1:
                batch_resized = batch_resized.repeat(1, 3, 1, 1)
            
            # Get predictions and convert to probabilities
            batch_predictions = model(batch_resized)
            batch_predictions = F.softmax(batch_predictions, dim=1)
            predictions.append(batch_predictions.cpu().numpy())
            
            processed_count += len(batch)
            if (processed_count // batch_size) % 20 == 0 and processed_count > 0:
                print(f"  Processed {processed_count}/{total_images} images...")
    
    print(f"  Processed {processed_count}/{total_images} images... Done.")
    predictions_array = np.concatenate(predictions, axis=0)
    print(f"  Predictions array shape: {predictions_array.shape}")
    
    return predictions_array


def calculate_inception_score(fake_images, device, batch_size=50, splits=10, return_probs=False):
    """
    Calculate Inception Score for generated images
    
    Args:
        fake_images: Generated images tensor
        device: Device to run on
        batch_size: Batch size for processing
        splits: Number of splits for confidence interval calculation
        return_probs: Whether to return class probabilities for visualization
    
    Returns:
        is_score: Mean Inception Score
        is_std: Standard deviation across splits
        predictions: Class probabilities (if return_probs=True)
    """
    print("\n" + "="*60 + "\nCOMPUTING INCEPTION SCORE\n" + "="*60)
    
    # Load classifier version of Inception
    inception_model = get_inception_model(device, return_features=False)
    
    # Get predictions for all fake images
    predictions = extract_predictions(fake_images, inception_model, device, batch_size)
    
    # Calculate IS for different splits
    print(f"Calculating IS with {splits} splits...")
    is_scores = []
    
    split_size = len(predictions) // splits
    for i in range(splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < splits - 1 else len(predictions)
        split_predictions = predictions[start_idx:end_idx]
        
        # Calculate marginal distribution p(y)
        marginal = np.mean(split_predictions, axis=0)
        
        # Calculate KL divergences
        kl_scores = []
        for pred in split_predictions:
            # Add small epsilon to prevent log(0)
            kl = np.sum(pred * np.log(pred / (marginal + 1e-16) + 1e-16))
            kl_scores.append(kl)
        
        # Calculate IS for this split
        mean_kl = np.mean(kl_scores)
        is_score = np.exp(mean_kl)
        is_scores.append(is_score)
    
    final_is_score = np.mean(is_scores)
    is_std = np.std(is_scores)
    
    print(f"Inception Score: {final_is_score:.3f} ± {is_std:.3f}")
    
    if return_probs:
        return final_is_score, is_std, predictions
    else:
        return final_is_score, is_std


def calculate_precision_recall(real_features, fake_features, k=3):
    """
    Calculate Precision and Recall using k-nearest neighbors
    
    Args:
        real_features: Features from real images (N, feature_dim)
        fake_features: Features from fake images (M, feature_dim)
        k: Number of nearest neighbors to consider
    
    Returns:
        precision: Fraction of fake images that are realistic
        recall: Fraction of real data manifold covered by fake images
        threshold: Distance threshold used
    """
    print("\n" + "="*60 + "\nCOMPUTING PRECISION/RECALL\n" + "="*60)
    print(f"Real features: {real_features.shape}")
    print(f"Fake features: {fake_features.shape}")
    print(f"k-nearest neighbors: {k}")
    
    # Calculate pairwise distances
    print("Computing pairwise distances...")
    fake_to_real_distances = pairwise_distances(fake_features, real_features, metric='euclidean')
    real_to_fake_distances = pairwise_distances(real_features, fake_features, metric='euclidean')
    
    print(f"  Fake→Real distances: {fake_to_real_distances.shape}")
    print(f"  Real→Fake distances: {real_to_fake_distances.shape}")
    
    # Calculate threshold based on real→fake k-nearest distances
    print("Determining threshold...")
    real_kth_distances = []
    for real_idx in range(len(real_to_fake_distances)):
        distances = real_to_fake_distances[real_idx]
        kth_distance = np.sort(distances)[k-1]  # k-th nearest (0-indexed)
        real_kth_distances.append(kth_distance)
    
    threshold = np.median(real_kth_distances)
    print(f"  Threshold: {threshold:.4f}")
    
    # Calculate Precision
    print("Calculating Precision...")
    realistic_count = 0
    for fake_idx in range(len(fake_to_real_distances)):
        min_distance_to_real = np.min(fake_to_real_distances[fake_idx])
        if min_distance_to_real <= threshold:
            realistic_count += 1
    
    precision = realistic_count / len(fake_features)
    print(f"  Realistic fake images: {realistic_count}/{len(fake_features)}")
    print(f"  Precision: {precision:.4f}")
    
    # Calculate Recall
    print("Calculating Recall...")
    covered_count = 0
    for real_idx in range(len(real_to_fake_distances)):
        min_distance_to_fake = np.min(real_to_fake_distances[real_idx])
        if min_distance_to_fake <= threshold:
            covered_count += 1
    
    recall = covered_count / len(real_features)
    print(f"  Covered real images: {covered_count}/{len(real_features)}")
    print(f"  Recall: {recall:.4f}")
    
    return precision, recall, threshold


def load_trained_generator(model_path, config_path, device):
    """Load your trained generator model using the project's factory."""
    print(f"Loading generator from {model_path}")
    
    with open(config_path, 'r') as f:
        data = json.load(f)
        arch_config = data.get('best_architecture', data)

    print("Creating model structure from config using factory...")
    generator, _ = create_models_from_config(arch_config, device)
    
    try:
        print("Loading trained weights...")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        generator.load_state_dict(state_dict)
        print("Generator loaded successfully")
    except Exception as e:
        print(f"Error loading generator state_dict: {e}")
        return None, None
    
    generator = generator.to(device)
    generator.eval()
    
    return generator, arch_config


def generate_images(generator, arch_config, num_samples, device, batch_size=100):
    """Generate fake images using the trained generator"""
    print(f"\n{'='*60}")
    print(f"GENERATING FAKE IMAGES")
    print(f"{'='*60}")
    print(f"Number of samples: {num_samples}")
    
    latent_dim = arch_config['generator']['latent_dim']
    print(f"Latent dimension: {latent_dim}")
    
    fake_images = []
    label_counts = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            noise = torch.randn(current_batch_size, latent_dim, device=device)
            labels = torch.randint(0, 10, (current_batch_size,), device=device)
            
            # Count labels for distribution check
            for label in labels:
                label_counts[label.item()] += 1
            
            fake_batch = generator(noise, labels)
            fake_images.append(fake_batch.cpu())
            
            if (len(fake_images) * batch_size) % (batch_size * 10) == 0:
                print(f"  Generated {min(len(fake_images) * batch_size, num_samples)}/{num_samples} images...")
    
    fake_images = torch.cat(fake_images, dim=0)[:num_samples]  # Ensure exact count
    
    print(f"Generated {len(fake_images)} fake images")
    print(f"Generated label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"   Digit {label}: {count} images")
    print(f"{'='*60}\n")
    
    return fake_images


def load_real_images_from_directory(data_dir, num_samples, device='auto'):
    """Load real images directly from data directory structure (without CSV)."""
    print(f"\n{'='*60}")
    print(f"LOADING REAL IMAGES FROM DIRECTORY")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Max samples: {num_samples}")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Match training normalization
    ])
    
    # Look for image files in the directory structure
    data_path = Path(data_dir)
    image_files = []
    
    # Search for common image extensions
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    for ext in extensions:
        image_files.extend(list(data_path.rglob(ext)))
    
    if not image_files:
        print(f"ERROR: No image files found in {data_dir}")
        return None
    
    print(f"Found {len(image_files)} image files")
    
    # Randomly sample images if we have more than needed
    if len(image_files) > num_samples:
        import random
        random.shuffle(image_files)
        image_files = image_files[:num_samples]
    
    print(f"Using {len(image_files)} images for evaluation")
    
    # Create dataset from file paths
    class DirectoryImageDataset(Dataset):
        def __init__(self, image_files, transform=None):
            self.image_files = image_files
            self.transform = transform
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            try:
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                if self.transform:
                    image = self.transform(image)
                return image
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Return a zero tensor as fallback
                return torch.zeros(1, 32, 32)
    
    dataset = DirectoryImageDataset(image_files, transform=transform)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=100, 
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    
    print(f"Created DataLoader with {len(dataset)} images")
    return dataloader


def load_real_images(data_dir, labels_csv, num_samples, split=None, device='auto'):
    """Load real images from your dataset using the CSV file."""
    print(f"\n{'='*60}")
    print(f"LOADING REAL IMAGES")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Labels CSV: {labels_csv}")
    print(f"Split: {split}")
    print(f"Max samples: {num_samples}")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Match training normalization
    ])
    
    dataset = FIDDataset(
        data_dir, 
        labels_csv, 
        transform=transform, 
        max_samples=num_samples,
        split=split
    )
    
    if len(dataset) == 0:
        print(f"ERROR: No images were loaded!")
        return None

    dataloader = DataLoader(
        dataset, 
        batch_size=100, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=(device == 'cuda')
    )
    
    print(f"Created DataLoader with {len(dataset)} real images")
    print(f"{'='*60}\n")
    
    return dataloader


def compute_fid_score(real_dataloader, fake_images, device):
    """Complete FID computation"""
    print("\n" + "="*60 + "\nCOMPUTING FID SCORE\n" + "="*60)
    
    inception_model = get_inception_model(device, return_features=True)
    
    print("\n1. Processing real images...")
    real_features = extract_features(real_dataloader, inception_model, device, is_real=True)
    
    print("\n2. Processing fake images...")
    fake_features = extract_features(fake_images, inception_model, device, is_real=False)
    
    print("\n3. Calculating statistics...")
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_fake, sigma_fake = calculate_statistics(fake_features)
    
    print("\n4. Computing FID score...")
    fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid_score, real_features, fake_features


def compute_comprehensive_metrics(real_dataloader, fake_images, device):
    """Compute FID, IS, Precision, and Recall"""
    print("\n" + "="*80)
    print("COMPREHENSIVE GAN EVALUATION METRICS")
    print("="*80)
    
    # 1. Compute FID and extract features
    fid_score, real_features, fake_features = compute_fid_score(real_dataloader, fake_images, device)
    
    # 2. Compute Inception Score and get probabilities for visualization
    is_score, is_std, fake_probabilities = calculate_inception_score(fake_images, device, return_probs=True)
    
    # 3. Compute Precision and Recall
    precision, recall, threshold = calculate_precision_recall(real_features, fake_features)
    
    return {
        'fid_score': fid_score,
        'is_score': is_score,
        'is_std': is_std,
        'precision': precision,
        'recall': recall,
        'threshold': threshold,
        'real_features': real_features,
        'fake_features': fake_features,
        'fake_probabilities': fake_probabilities
    }


def interpret_fid_score(fid_score):
    """Provide interpretation of the FID score"""
    print("\n" + "="*60 + "\nFID SCORE INTERPRETATION\n" + "="*60)
    
    if fid_score < 10: interpretation, quality = "EXCELLENT! Very realistic images", "Excellent"
    elif fid_score < 30: interpretation, quality = "GOOD! High quality images", "Good"
    elif fid_score < 80: interpretation, quality = "MODERATE quality images", "Moderate"
    elif fid_score < 150: interpretation, quality = "POOR quality images", "Poor"
    else: interpretation, quality = "VERY POOR quality images", "Very Poor"
    
    print(f"FID Score: {fid_score:.2f}")
    print(f"Quality: {quality}")
    print(f"Assessment: {interpretation}")
    print(f"\nFor reference (MNIST/grayscale datasets):")
    print(f"  - FID < 10:  Excellent results")
    print(f"  - FID 10-30: Good, usable results")
    print(f"  - FID 30-80: Moderate quality")
    print(f"  - FID > 100: Poor quality, likely mode collapse or training issues")
    
    if fid_score > 100:
        print(f"\nHIGH FID SCORE DETECTED!")
        print(f"Possible causes:")
        print(f"  1. Mode collapse - generator producing limited variety")
        print(f"  2. Training instability - generator not fully converged")
        print(f"  3. Architecture mismatch - model capacity too small")
        print(f"  4. Normalization issues - check image preprocessing")
        print(f"  5. Limited training data - model hasn't seen enough examples")


def interpret_comprehensive_metrics(metrics):
    """Provide comprehensive interpretation of all metrics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION INTERPRETATION")
    print("="*80)
    
    fid = metrics['fid_score']
    is_score = metrics['is_score']
    precision = metrics['precision']
    recall = metrics['recall']
    
    print(f"SUMMARY OF RESULTS:")
    print(f"   FID Score:      {fid:.2f}")
    print(f"   Inception Score: {is_score:.3f} ± {metrics['is_std']:.3f}")
    print(f"   Precision:      {precision:.3f}")
    print(f"   Recall:         {recall:.3f}")
    print(f"   Threshold:      {metrics['threshold']:.4f}")
    
    # Diagnostic analysis
    print(f"\nDIAGNOSTIC ANALYSIS:")
    
    if fid < 30 and is_score > 6 and precision > 0.7 and recall > 0.7:
        diagnosis = "EXCELLENT: High-quality generation with good diversity"
        recommendations = ["Model is performing well", "Consider fine-tuning for even better results"]
        
    elif precision > 0.8 and recall < 0.4:
        diagnosis = "MODE COLLAPSE: High quality but limited diversity"
        recommendations = [
            "Increase generator capacity", 
            "Adjust loss function to encourage diversity",
            "Try different NAS strategy (progressive vs multifidelity)",
            "Check if discriminator is too strong"
        ]
        
    elif precision < 0.4 and recall > 0.6:
        diagnosis = "POOR QUALITY: Good coverage but unrealistic images"
        recommendations = [
            "Improve training stability",
            "Check architecture capacity",
            "Verify data preprocessing and normalization",
            "Adjust learning rates"
        ]
        
    elif precision < 0.4 and recall < 0.4:
        diagnosis = "GENERAL FAILURE: Both quality and coverage issues"
        recommendations = [
            "Check training convergence",
            "Verify loss functions are working",
            "Ensure sufficient training data",
            "Try different architecture configurations"
        ]
        
    elif is_score < 3:
        diagnosis = "LOW INCEPTION SCORE: Quality or diversity issues"
        recommendations = [
            "Check if images are blurry or poorly formed",
            "Verify class balance in generated samples",
            "Consider longer training or different optimizer"
        ]
        
    else:
        diagnosis = "MODERATE: Mixed results with room for improvement"
        recommendations = [
            "Try different hyperparameters",
            "Compare with other NAS strategies",
            "Consider ensemble approaches"
        ]
    
    print(f"   {diagnosis}")
    print(f"\n   RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"   {rec}")
    
    # Specific guidance for your NAS-GAN project
    print(f"\n   FOR YOUR NAS-GAN PROJECT:")
    if fid > 100:
        print(f"        Very high FID suggests fundamental issues")
        print(f"        Compare progressive vs multifidelity vs random strategies")
        print(f"        Check if best_generator.pth vs generator_epoch_X.pth makes difference")
    
    if precision > 0.7 and recall < 0.3:
        print(f"      Focus on diversity: Your NAS might be finding architectures that")
        print(f"      excel at specific digits but fail to generalize to all 10 classes")
    
    return diagnosis


def create_comprehensive_visualizations(metrics, model_name, strategy_name,timestamp, 
                                      real_features=None, fake_features=None, 
                                      real_probabilities=None, fake_probabilities=None):
    """Create comprehensive visualizations for all evaluation metrics."""
    
    viz_dir = Path("experiments/evaluation/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Metrics Summary Dashboard (Top)
    ax1 = plt.subplot(4, 4, (1, 2))
    metrics_data = [metrics['fid_score'], metrics['is_score'], 
                   metrics['precision'], metrics['recall']]
    metrics_names = ['FID', 'IS', 'Precision', 'Recall']
    colors = ['red' if metrics['fid_score'] > 50 else 'orange' if metrics['fid_score'] > 30 else 'green',
              'green' if metrics['is_score'] > 6 else 'orange' if metrics['is_score'] > 3 else 'red',
              'green' if metrics['precision'] > 0.7 else 'orange' if metrics['precision'] > 0.4 else 'red',
              'green' if metrics['recall'] > 0.7 else 'orange' if metrics['recall'] > 0.4 else 'red']
    
    bars = ax1.bar(metrics_names, metrics_data, color=colors, alpha=0.7)
    ax1.set_title(f'Evaluation Metrics Overview\n{model_name} ({strategy_name})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. FID Distribution Visualization (Fréchet Distance)
    if real_features is not None and fake_features is not None:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from matplotlib.patches import Ellipse
            
            ax2 = plt.subplot(4, 4, (3, 4))
            
            # Apply PCA to reduce to 2D for visualization
            all_features = np.vstack([real_features[:1000], fake_features[:1000]])
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(all_features)
            
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(scaled_features)
            
            real_pca = pca_features[:len(real_features[:1000])]
            fake_pca = pca_features[len(real_features[:1000]):]
            
            # Calculate means and covariances in 2D space
            real_mean = np.mean(real_pca, axis=0)
            fake_mean = np.mean(fake_pca, axis=0)
            real_cov = np.cov(real_pca.T)
            fake_cov = np.cov(fake_pca.T)
            
            # Plot distributions as scatter points
            ax2.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.4, label='Real Images', 
                       s=15, c='blue', edgecolors='none')
            ax2.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.4, label='Generated Images', 
                       s=15, c='red', edgecolors='none')
            
            # Plot means
            ax2.scatter(real_mean[0], real_mean[1], s=200, c='darkblue', marker='x', 
                       linewidth=4, label=f'μ_real')
            ax2.scatter(fake_mean[0], fake_mean[1], s=200, c='darkred', marker='x', 
                       linewidth=4, label=f'μ_generated')
            
            # Draw covariance ellipses
            def draw_covariance_ellipse(mean, cov, ax, color, alpha=0.3):
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(eigenvals)
                ellipse = Ellipse(mean, width, height, angle=angle, 
                                facecolor=color, alpha=alpha, edgecolor=color, linewidth=2)
                ax.add_patch(ellipse)
                return ellipse
            
            draw_covariance_ellipse(real_mean, real_cov, ax2, 'blue', alpha=0.2)
            draw_covariance_ellipse(fake_mean, fake_cov, ax2, 'red', alpha=0.2)
            
            # Draw line between means
            ax2.plot([real_mean[0], fake_mean[0]], [real_mean[1], fake_mean[1]], 
                    'k--', linewidth=2, alpha=0.7, label='Distance between means')
            
            # Add distance annotation
            distance = np.linalg.norm(real_mean - fake_mean)
            mid_point = (real_mean + fake_mean) / 2
            ax2.annotate(f'||μ_r - μ_g|| = {distance:.2f}', 
                        xy=mid_point, xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                        fontsize=9, fontweight='bold')
            
            ax2.set_title('FID Distribution Comparison\n(Fréchet Distance in Feature Space)', fontweight='bold')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
        except ImportError:
            ax2 = plt.subplot(4, 4, (3, 4))
            ax2.text(0.5, 0.5, 'sklearn not available\nfor FID visualization', 
                    ha='center', va='center', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            ax2.set_title('FID Distribution Comparison', fontweight='bold')
    else:
        ax2 = plt.subplot(4, 4, (3, 4))
        ax2.text(0.5, 0.5, 'Feature data not available\nfor FID visualization', 
                ha='center', va='center', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        ax2.set_title('FID Distribution Comparison', fontweight='bold')
    
    # 3. FID Score Interpretation
    ax3 = plt.subplot(4, 4, 5)
    fid_ranges = ['Excellent\n(<30)', 'Good\n(30-50)', 'Fair\n(50-100)', 'Poor\n(>100)']
    fid_values = [30, 50, 100, 200]
    
    current_fid = metrics['fid_score']
    highlight_colors = ['lightgreen' if current_fid < 30 else 'lightgray',
                       'orange' if 30 <= current_fid < 50 else 'lightgray',
                       'yellow' if 50 <= current_fid < 100 else 'lightgray',
                       'lightcoral' if current_fid >= 100 else 'lightgray']
    
    bars = ax3.bar(fid_ranges, fid_values, color=highlight_colors, alpha=0.8)
    ax3.axhline(y=current_fid, color='black', linestyle='--', linewidth=2, label=f'Your FID: {current_fid:.1f}')
    ax3.set_title('FID Score Interpretation', fontweight='bold')
    ax3.set_ylabel('FID Score')
    ax3.legend()
    
    # 4. Inception Score Distribution
    ax4 = plt.subplot(4, 4, 6)
    if real_probabilities is not None and fake_probabilities is not None:
        # Plot entropy distributions
        real_entropy = -np.sum(real_probabilities * np.log(real_probabilities + 1e-8), axis=1)
        fake_entropy = -np.sum(fake_probabilities * np.log(fake_probabilities + 1e-8), axis=1)
        
        ax4.hist(real_entropy, bins=50, alpha=0.7, label='Real', color='blue', density=True)
        ax4.hist(fake_entropy, bins=50, alpha=0.7, label='Generated', color='red', density=True)
        ax4.axvline(np.mean(fake_entropy), color='red', linestyle='--', label=f'Generated Mean: {np.mean(fake_entropy):.2f}')
        ax4.set_title('Entropy Distribution (Lower = More Confident)', fontweight='bold')
        ax4.set_xlabel('Entropy')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # IS interpretation chart
        is_ranges = ['Poor\n(<3)', 'Fair\n(3-6)', 'Good\n(6-8)', 'Excellent\n(>8)']
        is_values = [3, 6, 8, 10]
        current_is = metrics['is_score']
        
        highlight_colors = ['lightcoral' if current_is < 3 else 'lightgray',
                           'yellow' if 3 <= current_is < 6 else 'lightgray',
                           'orange' if 6 <= current_is < 8 else 'lightgray',
                           'lightgreen' if current_is >= 8 else 'lightgray']
        
        bars = ax4.bar(is_ranges, is_values, color=highlight_colors, alpha=0.8)
        ax4.axhline(y=current_is, color='black', linestyle='--', linewidth=2, label=f'Your IS: {current_is:.2f}')
        ax4.set_title('Inception Score Interpretation', fontweight='bold')
        ax4.set_ylabel('IS Score')
        ax4.legend()
    
    # 5. Precision vs Recall Analysis
    ax5 = plt.subplot(4, 4, 7)
    
    # Create quadrant plot
    precision = metrics['precision']
    recall = metrics['recall']
    
    ax5.scatter(recall, precision, s=200, c='red', marker='o', edgecolors='black', linewidth=2, 
               label=f'Your Model\n(P:{precision:.3f}, R:{recall:.3f})')
    
    # Add quadrant labels and guidelines
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant annotations
    ax5.text(0.25, 0.75, 'High Precision\nLow Recall\n(Mode Collapse)', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax5.text(0.75, 0.75, 'High Precision\nHigh Recall\n(Excellent)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax5.text(0.25, 0.25, 'Low Precision\nLow Recall\n(Poor)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax5.text(0.75, 0.25, 'Low Precision\nHigh Recall\n(Unrealistic)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel('Recall (Coverage)')
    ax5.set_ylabel('Precision (Quality)')
    ax5.set_title('Precision vs Recall Analysis', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Feature Magnitude Distribution
    ax6 = plt.subplot(4, 4, 8)
    if real_features is not None and fake_features is not None:
        real_norms = np.linalg.norm(real_features, axis=1)
        fake_norms = np.linalg.norm(fake_features, axis=1)
        
        ax6.hist(real_norms, bins=50, alpha=0.7, label='Real', color='blue', density=True)
        ax6.hist(fake_norms, bins=50, alpha=0.7, label='Generated', color='red', density=True)
        ax6.axvline(np.mean(real_norms), color='blue', linestyle='--', label=f'Real Mean: {np.mean(real_norms):.2f}')
        ax6.axvline(np.mean(fake_norms), color='red', linestyle='--', label=f'Generated Mean: {np.mean(fake_norms):.2f}')
        ax6.set_title('Feature Magnitude Distribution', fontweight='bold')
        ax6.set_xlabel('L2 Norm')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Feature data not available\nfor this visualization', 
                ha='center', va='center', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        ax6.set_title('Feature Magnitude Distribution', fontweight='bold')
    
    # 7-8. Class Distribution Analysis (Bottom row)
    if fake_probabilities is not None:
        ax7 = plt.subplot(4, 4, (9, 10))
        
        # Class probability distribution
        class_probs = np.mean(fake_probabilities, axis=0)
        classes = np.arange(len(class_probs))
        
        bars = ax7.bar(classes, class_probs, alpha=0.7, color='skyblue')
        ax7.set_title('Generated Class Distribution', fontweight='bold')
        ax7.set_xlabel('Class (Digit)')
        ax7.set_ylabel('Average Probability')
        ax7.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, prob in zip(bars, class_probs):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 9-10. Distance Threshold Analysis
    ax8 = plt.subplot(4, 4, (11, 12))
    
    # Show threshold impact visualization
    threshold = metrics.get('threshold', 0.1)
    thresholds = np.linspace(0.01, 0.5, 50)
    
    # This would ideally show how precision/recall change with threshold
    # For now, show the selected threshold
    ax8.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Selected Threshold: {threshold:.3f}')
    ax8.set_xlim(0, 0.5)
    ax8.set_xlabel('Distance Threshold')
    ax8.set_ylabel('Metric Value')
    ax8.set_title('Threshold Selection Impact', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.text(0.5, 0.5, 'Threshold analysis\n(shows selected value)', 
            ha='center', va='center', transform=ax8.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    # 11-16. Diagnosis and Recommendations
    ax9 = plt.subplot(4, 4, (13, 16))
    ax9.axis('off')
    
    # Get diagnosis
    diagnosis = interpret_comprehensive_metrics(metrics)
    
    # Create text summary
    summary_text = f"""COMPREHENSIVE EVALUATION REPORT
Model: {model_name} | Strategy: {strategy_name}

METRICS SUMMARY:
• FID Score: {metrics['fid_score']:.2f}
• Inception Score: {metrics['is_score']:.3f} ± {metrics['is_std']:.3f}
• Precision: {metrics['precision']:.3f}
• Recall: {metrics['recall']:.3f}

DIAGNOSIS:
{diagnosis}

INTERPRETATION:
FID: Lower is better (<30=Excellent, 30-50=Good, >50=Poor)
IS: Higher is better (>6=Good, 3-6=Fair, <3=Poor)  
Precision: Quality (fraction of realistic generated images)
Recall: Coverage (fraction of real data modes captured)
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comprehensive visualization
    viz_filename = f"comprehensive_evaluation_{model_name}_{strategy_name}_{timestamp}.png"
    viz_path = viz_dir / viz_filename
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    
    print(f"Comprehensive visualization saved to: {viz_path}")
    
    # Also create individual metric plots
    create_individual_metric_plots(metrics, model_name, strategy_name, viz_dir,timestamp)
    
    return viz_path


def create_individual_metric_plots(metrics, model_name, strategy_name, viz_dir, timestamp):
    """Create individual detailed plots for each metric."""
    
    # 1. FID Comparison Chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Benchmark FID scores for context
    benchmarks = {
        'CIFAR-10 (Best)': 3.17,
        'CIFAR-10 (Good)': 10.0,
        'CelebA (Best)': 5.11,
        'Your Model': metrics['fid_score']
    }
    
    colors = ['green', 'lightgreen', 'orange', 'red' if metrics['fid_score'] > 50 else 'blue']
    bars = ax.bar(benchmarks.keys(), benchmarks.values(), color=colors, alpha=0.7)
    
    ax.set_title(f'FID Score Comparison\n{model_name} ({strategy_name})', fontsize=14, fontweight='bold')
    ax.set_ylabel('FID Score (Lower is Better)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, (name, value) in zip(bars, benchmarks.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(viz_dir / f"fid_comparison_{model_name}_{strategy_name}_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed FID Distribution Visualization
    if 'real_features' in metrics and 'fake_features' in metrics:
        create_detailed_fid_visualization(
            metrics['real_features'], 
            metrics['fake_features'], 
            metrics['fid_score'],
            model_name, 
            strategy_name, 
            viz_dir,
            timestamp,
        )
    
    # 3. Metrics Radar Chart
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics for radar chart
    categories = ['FID\n(inverted)', 'Inception\nScore', 'Precision', 'Recall']
    values = [
        1 - min(metrics['fid_score'] / 100, 1),  # Invert FID and cap at 100
        min(metrics['is_score'] / 10, 1),        # Cap IS at 10
        metrics['precision'],
        metrics['recall']
    ]
    
    # Close the radar chart
    values += values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=f'{model_name}')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(f'Model Performance Radar\n{model_name} ({strategy_name})', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(viz_dir / f"radar_chart_{model_name}_{strategy_name}_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_detailed_fid_visualization(real_features, fake_features, fid_score, model_name, strategy_name, viz_dir,timestamp):
    """Create a detailed FID calculation visualization similar to the Fréchet distance diagram."""
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from matplotlib.patches import Ellipse
        from scipy.linalg import sqrtm
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Main distribution plot (like your reference image)
        ax1 = plt.subplot(2, 3, (1, 2))
        
        # Apply PCA to reduce to 2D for visualization
        sample_size = min(2000, len(real_features), len(fake_features))
        real_sample = real_features[:sample_size]
        fake_sample = fake_features[:sample_size]
        
        all_features = np.vstack([real_sample, fake_sample])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(all_features)
        
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)
        
        real_pca = pca_features[:sample_size]
        fake_pca = pca_features[sample_size:]
        
        # Calculate statistics
        real_mean = np.mean(real_pca, axis=0)
        fake_mean = np.mean(fake_pca, axis=0)
        real_cov = np.cov(real_pca.T)
        fake_cov = np.cov(fake_pca.T)
        
        # Plot density-like visualization using histograms
        from scipy.stats import gaussian_kde
        
        # Create density plots
        xx, yy = np.mgrid[real_pca[:, 0].min():real_pca[:, 0].max():.1, 
                          real_pca[:, 1].min():real_pca[:, 1].max():.1]
        
        real_kde = gaussian_kde(real_pca.T)
        fake_kde = gaussian_kde(fake_pca.T)
        
        positions = np.vstack([xx.ravel(), yy.ravel()])
        real_density = np.reshape(real_kde(positions).T, xx.shape)
        fake_density = np.reshape(fake_kde(positions).T, xx.shape)
        
        # Plot contours (like the curves in your reference)
        ax1.contour(xx, yy, real_density, levels=5, colors='green', alpha=0.8, linewidths=2)
        ax1.contour(xx, yy, fake_density, levels=5, colors='blue', alpha=0.8, linewidths=2)
        
        # Fill areas
        ax1.contourf(xx, yy, real_density, levels=10, colors=['lightgreen'], alpha=0.3)
        ax1.contourf(xx, yy, fake_density, levels=10, colors=['lightblue'], alpha=0.3)
        
        # Mark means
        ax1.scatter(real_mean[0], real_mean[1], s=200, c='darkgreen', marker='x', 
                   linewidth=4, label='μ_real', zorder=10)
        ax1.scatter(fake_mean[0], fake_mean[1], s=200, c='darkblue', marker='x', 
                   linewidth=4, label='μ_generated', zorder=10)
        
        # Draw arrow between means
        ax1.annotate('', xy=fake_mean, xytext=real_mean,
                    arrowprops=dict(arrowstyle='<->', color='red', lw=3))
        
        # Add distance annotation
        distance = np.linalg.norm(real_mean - fake_mean)
        mid_point = (real_mean + fake_mean) / 2
        ax1.annotate(f'||μ_r - μ_g||² = {distance**2:.2f}', 
                    xy=mid_point, xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                    fontsize=12, fontweight='bold')
        
        ax1.set_title('Fréchet Distance Between Feature Distributions', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Principal Component 1', fontsize=12)
        ax1.set_ylabel('Principal Component 2', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. FID Formula breakdown
        ax2 = plt.subplot(2, 3, 3)
        ax2.axis('off')
        
        # Calculate actual FID components in full feature space
        real_mean_full = np.mean(real_features, axis=0)
        fake_mean_full = np.mean(fake_features, axis=0)
        real_cov_full = np.cov(real_features.T)
        fake_cov_full = np.cov(fake_features.T)
        
        mean_diff = np.linalg.norm(real_mean_full - fake_mean_full) ** 2
        trace_term = np.trace(real_cov_full + fake_cov_full - 2 * sqrtm(real_cov_full @ fake_cov_full))
        
        formula_text = f"""
FID CALCULATION BREAKDOWN

Formula:
FID = ||μᵣ - μₘ||² + Tr(Σᵣ + Σₘ - 2√(ΣᵣΣₘ))

Components:
• Mean difference: ||μᵣ - μₘ||² = {mean_diff:.2f}
• Covariance term: Tr(...) = {trace_term:.2f}

Final FID Score: {fid_score:.2f}

Interpretation:
• Lower is better
• < 30: Excellent
• 30-50: Good  
• 50-100: Fair
• > 100: Poor

Your score: {"Excellent" if fid_score < 30 else "Good" if fid_score < 50 else "Fair" if fid_score < 100 else "Poor"}
        """
        
        ax2.text(0.05, 0.95, formula_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        # 3. Component breakdown pie chart
        ax3 = plt.subplot(2, 3, 4)
        
        components = [mean_diff, trace_term]
        labels = [f'Mean Difference\n{mean_diff:.1f}', f'Covariance Term\n{trace_term:.1f}']
        colors = ['orange', 'skyblue']
        
        wedges, texts, autotexts = ax3.pie(components, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        ax3.set_title('FID Components Breakdown', fontweight='bold')
        
        # 4. Distribution comparison histograms
        ax4 = plt.subplot(2, 3, 5)
        
        # Project to 1D for histogram
        real_1d = real_pca @ np.array([1, 0])  # Project to first principal component
        fake_1d = fake_pca @ np.array([1, 0])
        
        ax4.hist(real_1d, bins=50, alpha=0.7, label='Real Images', color='green', density=True)
        ax4.hist(fake_1d, bins=50, alpha=0.7, label='Generated Images', color='blue', density=True)
        ax4.axvline(np.mean(real_1d), color='darkgreen', linestyle='--', linewidth=2, label='Real Mean')
        ax4.axvline(np.mean(fake_1d), color='darkblue', linestyle='--', linewidth=2, label='Generated Mean')
        
        ax4.set_title('1D Projection Comparison', fontweight='bold')
        ax4.set_xlabel('Feature Value (PC1)')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Score interpretation gauge
        ax5 = plt.subplot(2, 3, 6)
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        score_normalized = min(fid_score / 200, 1)  # Normalize to 0-1, cap at 200
        
        # Background sectors
        ax5.fill_between(theta, 0, 1, where=(theta <= np.pi * 0.15), color='green', alpha=0.3, label='Excellent (<30)')
        ax5.fill_between(theta, 0, 1, where=((theta > np.pi * 0.15) & (theta <= np.pi * 0.35)), color='orange', alpha=0.3, label='Good (30-50)')
        ax5.fill_between(theta, 0, 1, where=((theta > np.pi * 0.35) & (theta <= np.pi * 0.65)), color='yellow', alpha=0.3, label='Fair (50-100)')
        ax5.fill_between(theta, 0, 1, where=(theta > np.pi * 0.65), color='red', alpha=0.3, label='Poor (>100)')
        
        # Needle
        needle_angle = score_normalized * np.pi
        ax5.arrow(needle_angle, 0, 0, 0.8, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=3)
        
        ax5.set_xlim(0, np.pi)
        ax5.set_ylim(0, 1)
        ax5.set_title(f'FID Score Gauge\n{fid_score:.1f}', fontweight='bold', fontsize=14)
        ax5.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax5.set_xticklabels(['0', '50', '100', '150', '200'])
        ax5.set_xlabel('FID Score')
        
        plt.tight_layout()
        
        # Save the detailed FID visualization
        fid_viz_path = viz_dir / f"detailed_fid_analysis_{model_name}_{strategy_name}_{timestamp}.png"
        plt.savefig(fid_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed FID analysis saved to: {fid_viz_path}")
        
    except Exception as e:
        print(f"Could not create detailed FID visualization: {e}")



def main():
    
    parser = argparse.ArgumentParser(description='Calculate comprehensive GAN evaluation metrics (FID, IS, Precision, Recall)')
    
    parser.add_argument('--model', type=str, help='Path to trained generator model (.pth file)')
    parser.add_argument('--config', type=str, help='Path to architecture config (.json file)')
    parser.add_argument('--strategy', type=str, choices=['adaptive', 'progressive', 'multifidelity', 'random', 'adversarial', 'manual_dcgan'], help='NAS strategy to automatically find the latest model and config')
    parser.add_argument('--data-dir', type=str, default='data/DevanagariHandwrittenDigitDataset', help='Root path to the dataset directory')
    parser.add_argument('--labels-csv', type=str, default='data/hindi_mnist.csv', help='Path to the labels CSV for real images')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of images to use for evaluation (reduced default for faster computation)')
    parser.add_argument('--split', type=str, default=None, choices=['Train', 'Test', None], help='Which split to use for real images')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cuda/mps/cpu)')
    parser.add_argument('--use-directory', action='store_true', help='Load images directly from directory instead of using CSV')
    
    args = parser.parse_args()

    # Automatic path resolution if strategy is provided
    if args.strategy:
        print(f"[*] Strategy '{args.strategy}' provided. Attempting to auto-resolve paths...")
        
        # 1. Resolve Config Path
        nas_results_dir = Path("experiments/gan_run_models_and_images/nas_results")
        
        # Mapping for manual_dcgan baseline
        search_strategy_name = args.strategy
        if args.strategy == 'manual_dcgan':
            print("  [i] Manual DCGAN baseline detected. Using Manual DCGAN architecture.")
            search_strategy_name = 'manual_dcgan'

        potential_configs = [
            nas_results_dir / f"best_architecture_{search_strategy_name}.json",
            nas_results_dir / f"best_architecture_{search_strategy_name.replace('-', '_')}.json"
        ]
        
        resolved_config = None
        for p in potential_configs:
            if p.exists():
                resolved_config = str(p)
                break
        
        if not resolved_config:
            print(f"Error: Could not find architecture config for strategy '{args.strategy}' in {nas_results_dir}")
            sys.exit(1)
        
        if not args.config:
            args.config = resolved_config
            print(f"  [+] Resolved config: {args.config}")

        # 2. Resolve Model Path (find latest timestamped run)
        # Check both the standard NAS training root and the manual baseline root
        potential_training_roots = [
            Path("experiments/gan_run_models_and_images/final_training"),
            Path("experiments/manual_dcgan")
        ]
        
        latest_model = None
        latest_mtime = 0
        
        for root in potential_training_roots:
            if not root.exists(): continue
            
            # Match strategy name or folder (allowing for suffixes)
            strategy_dirs = list(root.glob(f"{args.strategy}*"))
            if not strategy_dirs and args.strategy == 'manual_dcgan':
                 # Fallback: check checkpoints folder inside manual_dcgan directly
                 strategy_dirs = [root]

            for s_dir in strategy_dirs:
                # Check for timestamped subdirs OR a direct 'checkpoints' folder
                runs = [d for d in s_dir.iterdir() if d.is_dir()]
                # Include the dir itself if it's a checkpoint dir
                runs.append(s_dir) 
                
                for run in runs:
                    # Check common checkpoint names/locations
                    potential_pth_files = [
                        run / "models" / "best_generator.pth",
                        run / "checkpoints" / "best_generator.pth",
                        run / "best_generator.pth"
                    ]
                    for model_pth in potential_pth_files:
                        if model_pth.exists():
                            mtime = model_pth.stat().st_mtime
                            if mtime > latest_mtime:
                                latest_mtime = mtime
                                latest_model = str(model_pth)
        
        if not latest_model:
            print(f"Error: Could not find any 'best_generator.pth' for strategy '{args.strategy}'")
            sys.exit(1)
            
        if not args.model:
            args.model = latest_model
            print(f"  [+] Resolved model: {args.model}")

    # Final validation
    if not args.model or not args.config:
        parser.error("The following arguments are required: --model and --config (unless --strategy is used)")


    device = resolve_device(args.device)
    
    print("="*80)
    print("COMPREHENSIVE GAN EVALUATION METRICS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Labels CSV: {args.labels_csv}")
    print(f"Split: {args.split}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Device: {device}")
    print(f"Metrics: FID, Inception Score, Precision, Recall")
    print("="*80)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = get_timestamp_from_path(args.model)
    
    try:
        # Load generator
        generator, arch_config = load_trained_generator(args.model, args.config, device)
        if generator is None:
            return
        
        # Load real images
        if args.use_directory:
            print("Using directory-based image loading...")
            real_dataloader = load_real_images_from_directory(args.data_dir, args.num_samples, device)
        else:
            print("Using CSV-based image loading...")
            real_dataloader = load_real_images(args.data_dir, args.labels_csv, args.num_samples, args.split, device)
        
        if real_dataloader is None:
            return
        
        # Generate fake images
        fake_images = generate_images(generator, arch_config, args.num_samples, device)
        
        # Get first batch of real images for comparison
        real_batch = next(iter(real_dataloader))
        
        # Save comparison grid
        strategy_name = get_strategy_from_path(args.model)
        if "manual" in str(args.model).lower():
            strategy_name = "manual_dcgan"
            
        model_name_stem = Path(args.model).stem
        save_comparison_grid(real_dataloader, fake_images, model_name_stem, strategy_name,timestamp)
        
        # Compute comprehensive metrics (FID, IS, Precision, Recall)
        metrics = compute_comprehensive_metrics(real_dataloader, fake_images, device)
        
        # Create comprehensive visualizations
        print("\nCreating comprehensive visualizations...")
        try:
            viz_path = create_comprehensive_visualizations(
                metrics, model_name_stem, strategy_name,timestamp,
                real_features=metrics.get('real_features'),
                fake_features=metrics.get('fake_features'),
                real_probabilities=metrics.get('real_probabilities'),
                fake_probabilities=metrics.get('fake_probabilities')
            )
        except Exception as viz_error:
            print(f"Warning: Could not create visualizations: {viz_error}")
            viz_path = None
        
        # Interpret results
        diagnosis = interpret_comprehensive_metrics(metrics)
        
        # Save results
        results = {
            'fid_score': float(metrics['fid_score']),
            'is_score': float(metrics['is_score']),
            'is_std': float(metrics['is_std']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'threshold': float(metrics['threshold']),
            'diagnosis': diagnosis,
            'model_path': args.model,
            'config_path': args.config,
            'num_samples': args.num_samples,
            'split': args.split,
            'strategy': strategy_name,
            'device': str(device)
        }
        
        results_dir = Path("experiments/evaluation/comprehensive_metrics")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_filename = f"comprehensive_metrics_{model_name_stem}_{strategy_name}_{timestamp}.json"
        results_file = results_dir / results_filename
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n Results saved to: {results_file}")
        print(f" Check the comparison images in: experiments/fid/fid_samples/")
        if viz_path:
            print(f" Check the comprehensive visualizations in: experiments/evaluation/visualizations/")
        print(f"\n QUICK SUMMARY:")
        print(f"  FID: {metrics['fid_score']:.2f} | IS: {metrics['is_score']:.3f} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f}")
        print(f"  Assessment: {diagnosis}")
        print(f"\n VISUALIZATION FILES:")
        if viz_path:
            print(f"  • Comprehensive Dashboard: {viz_path}")
            print(f"  • Individual Metrics: experiments/evaluation/visualizations/")
        print(f"  • Sample Comparisons: experiments/fid/fid_samples/")
        
    except Exception as e:
        print(f"\n An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()