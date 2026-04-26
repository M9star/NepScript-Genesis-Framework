"""
Data Utilities

This module provides utilities for loading and preprocessing the Devanagari
handwritten digit dataset for training conditional GANs.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np

class DevanagariDataset(Dataset):
    """Dataset class for Devanagari handwritten digits"""
    
    def __init__(self, csv_file, root_dir, split=None, transform=None):
        """
        Args:
            csv_file: Path to CSV file with image paths and labels
            root_dir: Root directory of the dataset
            split: 'Train' or 'Test'
            transform: Optional transforms to apply
        """
        self.data = pd.read_csv(csv_file)
        # self.data = self.data[self.data['filename'].str.contains(split)]
        if split is not None: 
            self.data = self.data[self.data['filename'].str.contains(split)]
        
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_rel_path = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        label = int(self.data.iloc[idx]['label'])
        
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_default_transform(image_size=32, augment=False):
    """Get default image transformation pipeline"""
    transforms_list = [
        transforms.Resize((image_size, image_size)),
    ]
    
    # Add data augmentation for training (OPTIMIZED FOR SPEED)
    if augment:
        transforms_list.extend([
            # Simplified augmentations - faster while maintaining diversity
            transforms.RandomRotation(degrees=10, fill=0),  # Reduced from 15° - faster
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),    # Keep translation
                scale=(0.95, 1.05),      # Reduced scale range - faster
                fill=0                   # Black background (matches digit ink)
            ),
            # Removed ColorJitter - CPU intensive, minimal benefit for grayscale
            # Removed GaussianBlur - CPU intensive, rarely applied (p=0.1)
        ])
    
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    return transforms.Compose(transforms_list)


def stratified_subset(dataset, max_per_class=None):
    """
    Create a stratified subset of the dataset
    
    Args:
        dataset: Source dataset
        max_per_class: Maximum samples per class
        
    Returns:
        Subset: Stratified subset of the dataset
    """
    # Get all labels
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    labels = np.array(labels)
    
    # Get indices for each class
    indices = []
    for class_id in range(10):
        class_indices = np.where(labels == class_id)[0]
        # Randomly sample up to max_per_class
        if len(class_indices) > max_per_class:
            class_indices = np.random.choice(class_indices, max_per_class, replace=False)
        indices.extend(class_indices.tolist())
    
    return Subset(dataset, indices)


def load_data(data_dir, labels_csv, batch_size=32, max_subset_per_class=None, split=None, num_workers=4, augment=False, device='cpu'):
    """
    Load and prepare the Devanagari digit dataset

    Args:
        data_dir: Path to dataset directory
        labels_csv: Path to labels CSV file
        batch_size: Batch size for DataLoader
        max_subset_per_class: Maximum samples per class (None for full dataset)
        split: Data split to use ('Train', 'Test', or None). Defaults to None.
        num_workers: Number of workers for DataLoader
        augment: Whether to apply data augmentation (recommended for training)
        
    Returns:
        tuple: (train_loader, dataset_size)
    """ 
    
    
    try:
        transform = get_default_transform(augment=augment)
        
        print(f"Data augmentation: {'Enabled' if augment else 'Disabled'}")
        
        train_dataset = DevanagariDataset(
            csv_file=labels_csv,
            root_dir=data_dir,
            # split='Train',                 #commented this line to use all the 20k images for training
            split=split,
            transform=transform
        )
        
        # Use stratified subset if specified
        if max_subset_per_class is not None and max_subset_per_class>0:
            print(f"creating a stratified subset with {max_subset_per_class} samples per class....")
            train_subset = stratified_subset(train_dataset, max_per_class=max_subset_per_class)
            data_source = train_subset
        else:
            print("Using the full dataset....")
            data_source = train_dataset
        
        # pin_memory speeds up CUDA transfers but is NOT supported on MPS
        use_pin_memory = (device == 'cuda') and torch.cuda.is_available()
        train_loader = DataLoader(
            data_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
        dataset_size = len(data_source)
        
        
        print(f"Dataset loaded: {dataset_size} samples using split = '{split}'")
        print(f"Batches per epoch: {len(train_loader)}")
        
        return train_loader, dataset_size
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the dataset is properly configured")
        return None, 0
