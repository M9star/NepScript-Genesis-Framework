"""
CNN Training Script for Devanagari Digit Classification

Usage:
    python scripts/train_cnn.py --epochs 100 --batch-size 64 --lr 0.001
    
    or with config:
    python scripts/train_cnn.py --config configs/cnn_training.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from nepscript.models.cnn import CNNClassifier
from nepscript.utils.data import DevanagariDataset
from nepscript.utils.config import get_timestamp, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN for digit classification')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, mps, cpu, or auto')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='experiments/cnn_models', 
                       help='Output directory for models')
    parser.add_argument('--dataset-dir', type=str, 
                       default='data/DevanagariHandwrittenDigitDataset',
                       help='Path to dataset directory')
    parser.add_argument('--csv-file', type=str, default='data/hindi_mnist.csv',
                       help='Path to CSV file with image metadata')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--early-stopping', type=int, default=10, 
                       help='Early stopping patience (0 to disable)')
    
    return parser.parse_args()


def load_datasets(csv_file, dataset_dir, batch_size, val_split=0.2):
    """Load train and test datasets with proper transforms."""
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)  # Normalize to [-1, 1]
    ])
    
    augment_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    # Load train and test splits
    train_dataset = DevanagariDataset(csv_file, dataset_dir, split='Train', 
                                     transform=augment_transform)
    test_dataset = DevanagariDataset(csv_file, dataset_dir, split='Test', 
                                    transform=transform)
    
    # Split train into train and validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def plot_history(history, output_dir):
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = output_dir / f'loss_curve_{get_timestamp()}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved loss curve to {plot_path}")


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_loader, val_loader, test_loader = load_datasets(
        args.csv_file, args.dataset_dir, args.batch_size, args.val_split
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")
    
    # Initialize model
    model = CNNClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_path = output_dir / 'best_cnn.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Saved best model to {best_model_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"Early stopping at epoch {epoch+1} (best: epoch {best_epoch+1})")
            break
    
    # Save training history
    history_path = output_dir / f'training_history_{get_timestamp()}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot loss curves
    plot_history(history, output_dir)

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()