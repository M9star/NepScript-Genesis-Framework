
"""
Image Generation Script

Script to generate synthetic Nepali digit images using a trained generator model.
It supports generating specific classes or random samples.

Usage:
example 1: use the best generator found by random search to generate 100 random samples of images: 

    python scripts/generate.py --model experiments/gan_run_models_and_images/final_training/random/models/best_generator.pth --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_random.json --num-samples 100


example 2: generate 50 samples of class 5 in a grid layout 
   
    python scripts/generate.py --model experiments/gan_run_models_and_images/final_training/random/models/best_generator.pth --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_random.json --class 5 --num-samples 50 --grid

change the --model and --arch-config to select from which model you want to generate the images
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))


from nepscript.models.factory import create_models_from_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic Nepali digit images using trained GAN'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained generator model (.pth file)'
    )
    
    parser.add_argument(
        '--arch-config',
        type=str,
        required=True,
        help='Path to architecture configuration JSON file'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to generate'
    )
    
    parser.add_argument(
        '--class',
        type=int,
        dest='target_class',
        help='Specific digit class to generate (0-9). If not specified, generates all classes'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/generated_samples',
        help='Directory to save generated images'
    )
    
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Save as a grid image instead of individual images'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use for generation'
    )
    
    return parser.parse_args()


def generate_samples(generator, latent_dim, num_samples, target_class=None, device='cuda'):
    """
    Generate samples using the trained generator
    
    Args:
        generator: Trained generator model
        latent_dim: Dimension of latent space
        num_samples: Number of samples to generate
        target_class: Specific class to generate (None for random)
        device: Device to use
        
    Returns:
        tuple: (generated_images, labels)
    """
    generator.eval()
    with torch.no_grad():
        # Generate noise
        noise = torch.randn(num_samples, latent_dim).to(device)
        
        # Generate labels
        if target_class is not None:
            labels = torch.full((num_samples,), target_class, dtype=torch.long).to(device)
        else:
            labels = torch.randint(0, 10, (num_samples,)).to(device)
        
        # Generate images
        fake_images = generator(noise, labels)
    
    return fake_images, labels


def save_as_grid(images, save_path, nrow=10):
    """Save images as a grid"""
    grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
    save_image(grid, save_path)
    print(f" Grid image saved to: {save_path}")
    
    # Also save a version with a title for easier viewing
    plot_image_path = save_path.with_name(f"{save_path.stem}_plot.png")
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f'Generated Nepali Digits ({len(images)} samples)')
    plt.tight_layout()
    plt.savefig(plot_image_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Grid plot saved to: {plot_image_path}")
    return save_path

def save_individual_images(images, labels, output_dir, arch_config, model_name):
    """Save images as individual files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique identifier for this generation run
    arch_id = f"{arch_config.get('id', 'unknown')}_{arch_config.get('sampling_strategy', 'random')}"
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        # Create class-specific subdirectory that includes the model name
        class_dir = output_dir / f"class_{label.item()}_{arch_id}_{model_name}"
        class_dir.mkdir(exist_ok=True)
        
        # Save image
        img_path = class_dir / f"sample_{idx:04d}.png"
        save_image(img, img_path, normalize=True)
    
    print(f" {len(images)} individual images saved to unique subdirectories in: {output_dir}")


def main():
    """Main execution function"""
    args = parse_args()
    
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("!!!  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load architecture configuration
    print(f"\n Loading architecture from: {args.arch_config}")
    with open(args.arch_config, 'r') as f:
        arch_config = json.load(f)
    
    nas_method = arch_config.get('sampling_strategy', 'random')
    arch_config['sampling_strategy'] = nas_method
    print(f"Detected NAS method: '{nas_method}'")
    
    # Create generator
    print("  Creating generator model...")
    generator, _ = create_models_from_config(arch_config, device)
    
    # Load trained weights
    print(f" Loading trained weights from: {args.model}")
    generator.load_state_dict(torch.load(args.model, map_location=device))
    generator.eval()
    
    # --- CHANGE: Get model name for unique file naming ---
    model_name = Path(args.model).stem
    print(f"   Using model name: '{model_name}' for file naming.")
    
    print(f" Model loaded successfully!")
    
    # Get latent dimension
    latent_dim = arch_config['generator']['latent_dim']
    
    # Generate samples
    print(f"\n Generating {args.num_samples} samples...")
    if args.target_class is not None:
        print(f"   Generating digit class: {args.target_class}")
    else:
        print("   Generating random classes")
    
    fake_images, labels = generate_samples(
        generator,
        latent_dim,
        args.num_samples,
        args.target_class,
        device
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- CHANGE: Create a unique identifier string ---
    arch_id = f"{arch_config.get('id', 'unknown')}_{arch_config.get('sampling_strategy', 'random')}"
    
    if args.grid:
        # --- CHANGE: Include model_name in the grid filename ---
        class_label = f"class_{args.target_class}" if args.target_class is not None else "mixed"
        grid_filename = f"generated_grid_{class_label}_{arch_id}_{model_name}_{args.num_samples}_samples.png"
        grid_path = output_dir / grid_filename
        save_as_grid(fake_images.cpu(), grid_path)
    else:
        # --- CHANGE: Pass model_name to the save function ---
        save_individual_images(fake_images.cpu(), labels.cpu(), output_dir, arch_config, model_name)
    
    # Print summary
    print("\n" + "="*60)
    print(" GENERATION COMPLETED!")
    print("="*60)
    print(f" Summary:")
    print(f"   Total samples: {args.num_samples}")
    if args.target_class is not None:
        print(f"   Class: {args.target_class}")
    else:
        print(f"   Classes: Mixed (0-9)")
    print(f"   Output directory: {output_dir}")
    print(f"   Format: {'Grid' if args.grid else 'Individual images'}")
    
    # Class distribution
    if args.target_class is None:
        unique, counts = torch.unique(labels.cpu(), return_counts=True)
        print(f"\n   Class distribution:")
        for cls, count in zip(unique.tolist(), counts.tolist()):
            print(f"      Class {cls}: {count} samples")


if __name__ == '__main__':
    main()