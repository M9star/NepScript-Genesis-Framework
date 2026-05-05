"""
CNN Evaluation Script for Devanagari Digit Classification

Computes accuracy, precision, recall, F1-score, and per-class metrics.

Usage:
    python scripts/evaluate_cnn.py --model experiments/cnn_models/best_cnn.pth
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from nepscript.models.cnn import CNNClassifier
from nepscript.utils.data import DevanagariDataset
from nepscript.utils.config import get_timestamp, resolve_device
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CNN model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, mps, cpu, or auto')
    parser.add_argument('--dataset-dir', type=str, 
                       default='data/DevanagariHandwrittenDigitDataset',
                       help='Path to dataset directory')
    parser.add_argument('--csv-file', type=str, default='data/hindi_mnist.csv',
                       help='Path to CSV file with image metadata')
    parser.add_argument('--output-dir', type=str, default='experiments/cnn_evaluation',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Save confusion matrix plot')
    
    return parser.parse_args()


def load_test_dataset(csv_file, dataset_dir, batch_size):
    """Load test dataset."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    test_dataset = DevanagariDataset(csv_file, dataset_dir, split='Test', 
                                    transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(predictions, labels):
    """Compute comprehensive metrics."""
    metrics = {
        'overall': {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision_macro': float(precision_score(labels, predictions, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(labels, predictions, average='weighted', zero_division=0)),
            'recall_macro': float(recall_score(labels, predictions, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(labels, predictions, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0)),
        },
        'per_class': {}
    }
    
    # Per-class metrics (one-vs-rest approach)
    for digit in range(10):
        # Create binary labels: 1 if digit matches, 0 otherwise
        binary_labels = (labels == digit).astype(int)
        binary_preds = (predictions == digit).astype(int)
        
        # Only compute if this digit exists in test set
        if binary_labels.sum() > 0:
            digit_acc = accuracy_score(binary_labels, binary_preds)
            digit_precision = precision_score(binary_labels, binary_preds, 
                                             zero_division=0)
            digit_recall = recall_score(binary_labels, binary_preds, 
                                       zero_division=0)
            digit_f1 = f1_score(binary_labels, binary_preds, 
                               zero_division=0)
            
            metrics['per_class'][f'digit_{digit}'] = {
                'accuracy': float(digit_acc),
                'precision': float(digit_precision),
                'recall': float(digit_recall),
                'f1_score': float(digit_f1),
                'samples': int(binary_labels.sum())
            }
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(labels, predictions, 
                                  target_names=[f'digit_{i}' for i in range(10)],
                                  output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    return metrics


def plot_confusion_matrix(cm, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Devanagari Digit Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  ✓ Confusion matrix saved to {output_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Setup
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = CNNClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Load test dataset
    print("Loading test dataset...")
    test_loader = load_test_dataset(args.csv_file, args.dataset_dir, args.batch_size)
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, labels = evaluate(model, test_loader, device)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(predictions, labels)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    for key, value in metrics['overall'].items():
        print(f"  {key:.<30} {value:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for digit, digit_metrics in metrics['per_class'].items():
        print(f"\n  {digit}:")
        for key, value in digit_metrics.items():
            if key != 'samples':
                print(f"    {key:.<25} {value:.4f}")
            else:
                print(f"    {key:.<25} {value}")
    
    # Save results
    results_path = output_dir / f'cnn_evaluation_{get_timestamp()}.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Save confusion matrix
    if args.visualize:
        cm_path = output_dir / f'confusion_matrix_{get_timestamp()}.png'
        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(cm, cm_path)
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
