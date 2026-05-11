"""
CNN Training Script for Devanagari Digit Classification

Usage:
    # Full original dataset
    python scripts/train_cnn.py --epochs 100 --batch-size 64

    # Low-resource baseline: 500 real samples per class only
    python scripts/train_cnn.py \
        --epochs 15 --batch-size 32 --early-stopping 0 \
        --samples-per-class 500 \
        --output-dir experiments/cnn_500_baseline

    # Low-resource + GAN: 500 real + 1000 GAN per class
    python scripts/train_cnn.py \
        --epochs 15 --batch-size 32 --early-stopping 0 \
        --samples-per-class 500 \
        --gan-dir data/synthetic_digits_10k \
        --gan-samples-per-class 1000 \
        --output-dir experiments/cnn_models_with_synthetic

NOTE:
    - Training uses real (+ GAN if provided) data only.
    - Validation split is taken from training data to monitor overfitting.
    - Test evaluation is done SEPARATELY via: python scripts/evaluate_cnn.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader, random_split, Subset, ConcatDataset, Dataset
)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from nepscript.models.cnn import CNNClassifier
from nepscript.utils.data import DevanagariDataset
from nepscript.utils.config import get_timestamp, resolve_device


# ── GAN Image Dataset ─────────────────────────────────────────────────────────

class GANImageDataset(Dataset):
    """
    Loads GAN-generated images.

    Supports folder names like:
        class_0_G9913_D4729_adaptive_pure_exploration_best_generator/
        class_1_G9913_D4729_adaptive_pure_exploration_best_generator/
        ...
        class_9_G9913_D4729_adaptive_pure_exploration_best_generator/

    The digit label is parsed from the FIRST integer (0-9) found in
    the folder name split by underscores.
    """

    def __init__(self, gan_dir, samples_per_class, transform=None):
        self.transform = transform
        self.samples = []
        gan_dir = Path(gan_dir)

        if not gan_dir.exists():
            raise FileNotFoundError(f"GAN directory not found: {gan_dir}")

        subdirs = sorted([d for d in gan_dir.iterdir() if d.is_dir()])

        if subdirs:
            print(f"  Found {len(subdirs)} subfolders in {gan_dir.name}/")
            for subdir in subdirs:
                label = self._parse_label_from_dirname(subdir.name)
                if label is None:
                    print(f"  Skipping folder (no digit found): {subdir.name}")
                    continue

                images = sorted(
                    list(subdir.glob('*.png')) +
                    list(subdir.glob('*.jpg')) +
                    list(subdir.glob('*.jpeg'))
                )
                chosen = images[:samples_per_class]
                for img_path in chosen:
                    self.samples.append((img_path, label))
                print(f"  GAN digit {label}: {len(chosen)} samples "
                      f"← {subdir.name}/")
        else:
            # Flat directory fallback
            all_images = sorted(
                list(gan_dir.glob('*.png')) + list(gan_dir.glob('*.jpg'))
            )
            print(f"  Flat directory: {len(all_images)} images found")
            class_files = {i: [] for i in range(10)}
            for img_path in all_images:
                label = self._parse_label_from_dirname(img_path.stem)
                if label is not None:
                    class_files[label].append(img_path)
            for label in range(10):
                chosen = class_files[label][:samples_per_class]
                for img_path in chosen:
                    self.samples.append((img_path, label))
                print(f"  GAN digit {label}: {len(chosen)} samples")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No GAN images found in {gan_dir}.\n"
                f"Folder structure detected:\n"
                f"  {[d.name for d in subdirs[:3]]}...\n"
                f"Make sure folder names contain the digit class "
                f"(e.g. class_0_..., digit_1_...)."
            )

        print(f"  Total GAN samples loaded: {len(self.samples)}")

    def _parse_label_from_dirname(self, name):
        """
        Extract digit label (0-9) from folder/file name.
        Scans underscore-separated tokens, returns first valid digit.

        Examples:
            'class_0_G9913_D4729_...'  → 0
            'digit_3_sample'           → 3
            '5_images'                 → 5
            'random_folder'            → None
        """
        parts = name.split('_')
        for part in parts:
            try:
                val = int(part)
                if 0 <= val <= 9:
                    return val
            except ValueError:
                continue
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        # always return label as LongTensor
        return image, torch.tensor(label, dtype=torch.long)


# ── TensorLabel Wrapper ───────────────────────────────────────────────────────

class TensorLabelWrapper(Dataset):
    """
    Wraps any dataset to ensure labels are always returned as LongTensors.

    This is needed because DevanagariDataset returns plain int labels,
    while GANImageDataset returns tensor labels. When mixed in a
    ConcatDataset, PyTorch's collator crashes on mismatched types.
    Wrapping both datasets fixes this before collation happens.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return image, label


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CNN for Devanagari digit classification'
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                    default='experiments/cnn_models' if True else '',
                    help='Where to save model checkpoints and logs')
    parser.add_argument('--dataset-dir', type=str,
                        default='data/DevanagariHandwrittenDigitDataset')
    parser.add_argument('--csv-file', type=str,
                        default='data/hindi_mnist.csv')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Fraction of training data for validation '
                             '(default: 0.1). Test set never used here.')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Patience epochs (0 = disabled)')

    # Limit real samples
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='Max real training samples per class '
                             '(e.g. 500). Default: use all.')

    # GAN augmentation
    parser.add_argument('--gan-dir', type=str, default=None,
                        help='Path to GAN-generated images directory.')
    parser.add_argument('--gan-samples-per-class', type=int, default=1000,
                        help='GAN samples per class to add (default: 1000)')

    return parser.parse_args()


# ── Dataset Loading ───────────────────────────────────────────────────────────

def load_train_dataset(csv_file, dataset_dir, val_split,
                       samples_per_class, gan_dir, gan_samples_per_class):
    """
    Builds training dataset:
      1. Load real Train split from DHCD
      2. Optionally subsample to samples_per_class per class
      3. Optionally concatenate GAN images
      4. Wrap both with TensorLabelWrapper to fix label type mismatch
      5. Split into train / val for monitoring only

    Test set is never loaded here — use evaluate_cnn.py for final numbers.
    """
    from torchvision import transforms

    augment_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    # ── Step 1: Real training data ────────────────────────────────────────────
    real_dataset = DevanagariDataset(
        csv_file, dataset_dir, split='Train',
        transform=augment_transform
    )
    print(f"  Real training set (full): {len(real_dataset)} samples")

    # ── Step 2: Subsample per class ───────────────────────────────────────────
    if samples_per_class is not None:
        print(f"\n  Subsampling to {samples_per_class} real samples/class...")
        class_indices = {i: [] for i in range(10)}
        for idx in range(len(real_dataset)):
            _, label = real_dataset[idx]
            class_indices[int(label)].append(idx)

        selected = []
        for digit in range(10):
            available = class_indices[digit]
            n = min(samples_per_class, len(available))
            chosen = np.random.choice(available, size=n, replace=False).tolist()
            selected.extend(chosen)
            print(f"    Digit {digit}: {n} real samples")

        real_dataset = Subset(real_dataset, selected)
        print(f"  Real samples after subsampling: {len(real_dataset)}")

    # ── Step 3: GAN augmentation ──────────────────────────────────────────────
    if gan_dir is not None:
        print(f"\n  Loading GAN images from: {gan_dir}")
        print(f"  GAN samples per class   : {gan_samples_per_class}")
        gan_dataset = GANImageDataset(
            gan_dir,
            samples_per_class=gan_samples_per_class,
            transform=augment_transform
        )
        # wrap both datasets so labels are always LongTensor before collation
        real_wrapped = TensorLabelWrapper(real_dataset)
        gan_wrapped  = TensorLabelWrapper(gan_dataset)
        combined     = ConcatDataset([real_wrapped, gan_wrapped])
        real_count   = len(real_dataset)
        gan_count    = len(gan_dataset)
        print(f"\n  Training data summary:")
        print(f"    Real samples : {real_count}")
        print(f"    GAN samples  : {gan_count}")
        print(f"    Total        : {len(combined)}")
    else:
        # wrap even without GAN to keep consistent label types
        combined = TensorLabelWrapper(real_dataset)
        print(f"\n  Training data summary:")
        print(f"    Real only    : {len(combined)} samples (no GAN)")

    # ── Step 4: Train / val split ─────────────────────────────────────────────
    total      = len(combined)
    val_size   = int(val_split * total)
    train_size = total - val_size
    train_sub, val_sub = random_split(combined, [train_size, val_size])

    print(f"\n  Train split : {train_size}")
    print(f"  Val split   : {val_size}")
    print(f"  (Run evaluate_cnn.py for final test accuracy)\n")

    return train_sub, val_sub


# ── Training & Validation ─────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'],   label='Val Acc')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = output_dir / f'loss_curve_{get_timestamp()}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓ Loss curve → {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Run mode label
    if args.gan_dir:
        mode = "with_synthetic"
    else:
        mode = "baseline"
    print(f"Run mode: {mode}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_sub, val_sub = load_train_dataset(
        args.csv_file, args.dataset_dir,
        args.val_split, args.samples_per_class,
        args.gan_dir, args.gan_samples_per_class
    )

    train_loader = DataLoader(train_sub, batch_size=args.batch_size,
                              shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=args.batch_size,
                              shuffle=False)

    # Model
    model     = CNNClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"Training for {args.epochs} epochs...")
    best_val_acc, best_epoch, patience_counter = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': [],
               'val_loss':   [], 'val_acc':   []}

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader,
                                      criterion, optimizer, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch+1:>3}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc     = vl_acc
            best_epoch       = epoch
            patience_counter = 0
            if args.gan_dir:
                save_path = output_dir / 'best_cnn_with_synthetic.pth'
            else:
                save_path = output_dir / 'best_cnn.pth'

            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model → {save_path}")
        else:
            patience_counter += 1

        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(best: epoch {best_epoch+1})")
            break

    # Save history
    history_path = output_dir / f'history_{get_timestamp()}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    plot_history(history, output_dir)

    print(f"\n{'='*55}")
    print(f"Mode         : {mode}")
    print(f"Best val acc : {best_val_acc:.4f}")
    print(f"Model saved  : {output_dir}/best_cnn_{mode}.pth")
    print(f"{'='*55}")
    print(f"\nNext — evaluate on test set:")
    print(f"  python scripts/evaluate_cnn.py "
          f"--model {output_dir}/best_cnn_{mode}.pth")


if __name__ == '__main__':
    main()