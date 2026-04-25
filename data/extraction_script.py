"""
Devanagari Hindi MNIST Dataset Extraction Script

This script downloads the Hindi MNIST dataset from Kaggle using kagglehub.
Run this script to automatically download and set up the dataset for training.

Requirements:
- pip install kagglehub
- Kaggle account (for authentication if needed)
"""

# Install dependencies as needed:
# pip install kagglehub
import kagglehub
import os
import shutil
from pathlib import Path

def download_hindi_mnist_dataset():
    """Download and extract Hindi MNIST dataset from Kaggle"""
    
    print("🔄 Downloading Hindi MNIST dataset from Kaggle...")
    print("📁 Dataset: anurags397/hindi-mnist-data")
    
    try:
        # Download the dataset using kagglehub
        path = kagglehub.dataset_download("anurags397/hindi-mnist-data")
        
        print(f"✅ Dataset downloaded to: {path}")
        
        # Current data directory
        current_dir = Path(__file__).parent
        target_dir = current_dir
        
        # Copy the downloaded files to our data directory
        downloaded_path = Path(path)
        
        print("📋 Copying files to data directory...")
        
        # Look for the dataset files in the downloaded directory
        for item in downloaded_path.rglob("*"):
            if item.is_file():
                # Calculate relative path and copy
                rel_path = item.relative_to(downloaded_path)
                target_file = target_dir / rel_path
                
                # Create parent directories if they don't exist
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(item, target_file)
                print(f"  ✓ Copied: {rel_path}")
        
        print("🎉 Dataset extraction completed successfully!")
        print("📂 Files are now available in the data/ directory")
        
        # List the contents to verify
        print("\n📋 Available files:")
        for item in sorted(target_dir.rglob("*")):
            if item.is_file() and item.name != "extraction_script.py":
                rel_path = item.relative_to(target_dir)
                print(f"  • {rel_path}")
                
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Make sure you have kagglehub installed: pip install kagglehub")
        print("  2. You might need to authenticate with Kaggle")
        print("  3. Check your internet connection")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("📦 HINDI MNIST DATASET DOWNLOADER")
    print("=" * 60)
    
    success = download_hindi_mnist_dataset()
    
    if success:
        print("\n✅ Setup complete! You can now run training scripts.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")