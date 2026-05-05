"""
Create CSV for Generated Synthetic Digits

Organizes generated images by digit class and creates combined CSV for training.

Usage:
    python scripts/create_synthetic_csv.py --generated-dir data/synthetic_digits
"""

import argparse
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Create CSV from generated synthetic images')
    parser.add_argument('--generated-dir', type=str, required=True,
                       help='Directory containing generated images')
    parser.add_argument('--original-csv', type=str, default='data/hindi_mnist.csv',
                       help='Path to original CSV')
    parser.add_argument('--output-csv', type=str, default='data/hindi_mnist_with_synthetic.csv',
                       help='Output CSV path')
    
    return parser.parse_args()


def organize_generated_images(generated_dir):
    """Organize generated images by digit class in data section."""
    generated_dir = Path(generated_dir)
    data_dir = Path('data/DevanagariHandwrittenDigitDataset')
    synthetic_dir = data_dir / 'Synthetic'
    
    print(f"\nOrganizing generated images...")
    print(f"Source: {generated_dir}")
    print(f"Destination: {synthetic_dir}")
    
    # Create synthetic directory structure
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    file_records = []
    
    # Find all generated images (they're in subdirectories by class)
    class_dirs = list(generated_dir.glob('class_*'))
    
    if not class_dirs:
        print(f"Warning: No 'class_*' directories found in {generated_dir}")
        print(f"Available directories: {list(generated_dir.glob('*'))}")
        return []
    
    for class_dir in sorted(class_dirs):
        # Extract digit from directory name (e.g., "class_5_adaptive_random" -> 5)
        try:
            digit = int(class_dir.name.split('_')[1])
        except (IndexError, ValueError):
            print(f"  Skipping {class_dir.name} - couldn't parse digit")
            continue
        
        digit_dir = synthetic_dir / f"Synthetic_digit_{digit}"
        digit_dir.mkdir(exist_ok=True)
        
        # Copy images from class directory to organized digit directory
        image_files = list(class_dir.glob('*.png'))
        
        print(f"Processing digit {digit}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"Digit {digit}", leave=False):
            # Create unique filename
            new_filename = f"synthetic_{digit}_{img_file.stem}.png"
            new_path = digit_dir / new_filename
            
            # Copy file
            shutil.copy2(img_file, new_path)
            
            # Record for CSV
            rel_path = f"DevanagariHandwrittenDigitDataset/Synthetic/Synthetic_digit_{digit}/{new_filename}"
            file_records.append({
                'filename': rel_path,
                'label': digit,
                'source': 'synthetic',
                'model': 'adaptive-gan'
            })
    
    print(f"✓ Organized {len(file_records)} synthetic images")
    return file_records


def create_combined_csv(original_csv, synthetic_records, output_csv):
    """Combine original and synthetic data into single CSV."""
    
    # Load original CSV
    original_df = pd.read_csv(original_csv)
    
    # Add source column if not present
    if 'source' not in original_df.columns:
        original_df['source'] = 'original'
    if 'model' not in original_df.columns:
        original_df['model'] = 'na'
    
    # Create synthetic dataframe
    synthetic_df = pd.DataFrame(synthetic_records)
    
    # Combine
    combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Combined CSV created: {output_csv}")
    print(f"{'='*60}")
    print(f"Original samples:  {len(original_df)}")
    print(f"Synthetic samples: {len(synthetic_df)}")
    print(f"Total samples:     {len(combined_df)}")
    
    # Breakdown by digit
    print(f"\nBreakdown by digit:")
    for digit in range(10):
        orig_count = len(original_df[original_df['label'] == digit])
        synth_count = len(synthetic_df[synthetic_df['label'] == digit])
        total = orig_count + synth_count
        print(f"  Digit {digit}: {orig_count:4d} (orig) + {synth_count:4d} (synth) = {total:4d} total")
    
    return combined_df


def main():
    args = parse_args()
    
    # Check if generated directory exists
    if not Path(args.generated_dir).exists():
        print(f"Error: Generated directory not found: {args.generated_dir}")
        print(f"Please run: python scripts/generate.py first")
        return
    
    # Organize generated images
    synthetic_records = organize_generated_images(args.generated_dir)
    
    if not synthetic_records:
        print("Error: No synthetic images were processed!")
        return
    
    # Create combined CSV
    combined_df = create_combined_csv(args.original_csv, synthetic_records, args.output_csv)
    
    print(f"\n✓ Ready to train CNN with combined data!")
    print(f"\nNext step:")
    print(f"python scripts/train_cnn.py --epochs 100 --csv-file {args.output_csv}")


if __name__ == '__main__':
    main()
