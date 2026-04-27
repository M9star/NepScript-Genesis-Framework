"""
GAN Training Script

Script to train a specific GAN architecture using the provided configuration.
It does comprehensive training with checkpointing saving.

Usage:

    to train the best_config using default training args: 

    adaptive :: python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_adaptive.json --epochs 400
    progressive :: python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_progressive.json --epochs 400
    multifidelity :: python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_multifidelity.json --epochs 400
    random :: python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_random.json --epochs 400

    train any other model you want : 
    
    python scripts/train_model.py --config configs/default_training.yaml --arch-config path/CONFIGURATION JSON YOU WANT TO TRAIN 

"""

import argparse
import json
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import re 

# Add project root to path
# --- Project Path Setup ---
project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from nepscript.models.factory import create_models_from_config, weights_init
from nepscript.training.trainer import GANTrainer
from nepscript.utils.data import load_data
from nepscript.utils.config import (
    load_config, validate_training_config, 
    get_default_training_config, merge_configs,
    resolve_device
)



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train a Conditional DCGAN with specified architecture'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to training configuration YAML file'
    )
    
    parser.add_argument(
        '--arch-config',
        type=str,
        help='Path to architecture configuration JSON file'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['adaptive', 'progressive', 'multifidelity', 'random', 'adversarial', 'manual_dcgan'],
        help='NAS strategy to automatically find the best architecture config'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        help='Path to a generator checkpoint (.pth file) to resume training from'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--g-lr',
        type=float,
        help='Generator learning rate (default: 0.0001)'
    )
    
    parser.add_argument(
        '--d-lr',
        type=float,
        help='Discriminator learning rate (default: 0.0002)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training (default: 64)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory to save model checkpoints'
    )
    
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory to save sample images'
    )
    
    parser.add_argument(
        '--save-every',
        type=int,
        help='Save full checkpoint every N epochs (default: 100)'
    )
    
    parser.add_argument(
        '--gen-save-every',
        type=int,
        help='Save generator checkpoint every N epochs (default: 10)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps', 'auto'],
        default='auto',
        help='Device to use for training (auto, mps, cuda, or cpu)'
    )
    parser.add_argument(
    '--max-subset-per-class',
    type=int,
    help='Maximum samples per class for stratified subset'
    )

    parser.add_argument(
        '--split',
        type=str,
        help='Train or Test or None'
    )
    
    parser.add_argument(
        '--d-update-freq',
        type=int,
        help='Update discriminator every N batches (default: 1)'
    )
    
    parser.add_argument(
    '--no-augment',
    action='store_true',
    help='Disable data augmentation (use original images only)'
    )

    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation (default behavior)'
    )
    
    return parser.parse_args()


def plot_training_curves(g_losses, d_losses, save_path):
    """Plot and save training loss curves"""
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.axhline(y=1.386, color='r', linestyle='--', label='Theoretical Equilibrium')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Training curves saved to: {save_path}")


def main():
    """Main execution function"""
    args = parse_args()
    
    # Automatic path resolution if strategy is provided
    if args.strategy:
        print(f"[*] Strategy '{args.strategy}' provided. Attempting to auto-resolve arch-config...")
        nas_results_dir = Path("experiments/gan_run_models_and_images/nas_results")
        
        # Mapping for manual_dcgan baseline
        search_strategy_name = args.strategy
        if args.strategy == 'manual_dcgan':
            print("  [i] Manual DCGAN baseline detected. Using Random Search winner as architecture.")
            search_strategy_name = 'random'
            # Also auto-load the manual training config if no config is provided
            if not args.config:
                manual_config = "configs/manual_dcgan_training.yaml"
                if Path(manual_config).exists():
                    args.config = manual_config
                    print(f"  [+] Resolved training config: {args.config}")

        # Check potential filenames
        potential_configs = [
            nas_results_dir / f"best_architecture_{search_strategy_name}.json",
            nas_results_dir / f"best_architecture_{search_strategy_name.replace('-', '_')}.json"
        ]
        
        resolved_path = None
        for p in potential_configs:
            if p.exists():
                resolved_path = str(p)
                break
        
        if resolved_path:
            if not args.arch_config:
                args.arch_config = resolved_path
                print(f"  [+] Resolved arch-config: {args.arch_config}")
        else:
            print(f"Error: Could not find architecture config for strategy '{args.strategy}' in {nas_results_dir}")
            sys.exit(1)

    if not args.arch_config:
        print("Error: The following argument is required: --arch-config (unless --strategy is used)")
        sys.exit(1)

    # Load architecture configuration
    print(f"\n Loading architecture from: {args.arch_config}")
    with open(args.arch_config, 'r') as f:
        arch_config = json.load(f)
    
    nas_method = arch_config.get('sampling_strategy', 'random')
    print(f"Detected NAS method:'{nas_method}'")
    
    config = get_default_training_config(nas_method) 
    
    if args.config:
        print(f"Loading base training configuration from: {args.config}")
        yaml_config = load_config(args.config)
        config = merge_configs(config, yaml_config)
    
    config = merge_configs(config, arch_config)
    
    overrides = {}
    if args.epochs:
        overrides['epochs'] = args.epochs
    if args.checkpoint_dir:
        overrides['checkpoint_dir'] = args.checkpoint_dir
    if args.image_dir:
        overrides['image_dir'] = args.image_dir
    if args.save_every:
        overrides['save_every'] = args.save_every
    if args.device:
        overrides['device'] = args.device
    if args.g_lr:
        overrides['g_lr'] = args.g_lr
    if args.d_lr:
        overrides['d_lr'] = args.d_lr
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    
    
    # Apply configuration overrides with safety checks
    if args.max_subset_per_class is not None or args.split is not None:
        overrides.setdefault('data',{})
    if args.max_subset_per_class is not None:
        overrides['data']['max_subset_per_class'] = args.max_subset_per_class
    if args.split is not None:
        overrides['data']['split'] = None if args.split.lower() == 'none' else args.split 
            
    if args.no_augment or args.augment:
        overrides.setdefault('data', {})
        if args.no_augment:
            overrides['data']['augment'] = False
        elif args.augment:
            overrides['data']['augment'] = True
        
        
    if args.d_update_freq:
        overrides['d_update_freq'] = args.d_update_freq
    
    if args.save_every:
        overrides['save_every'] = args.save_every
    if args.gen_save_every:
        overrides['gen_save_every'] = args.gen_save_every
    
    
    
    
    config = merge_configs(config, overrides)
    
    try:
        validate_training_config(config)
    except ValueError as e:
        print(f" Configuration error: {e}")
        sys.exit(1)
    
    device = resolve_device(config.get('device', 'auto'))
    
    print(f"Using device: {device}")
    
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', 'data/DevanagariHandwrittenDigitDataset')
    labels_csv = data_config.get('labels_csv', 'data/hindi_mnist.csv')
    max_subset_per_class = data_config.get('max_subset_per_class', None)
    split = data_config.get('split', None)
    augment = data_config.get('augment', True)  # Enable augmentation by default
    
    batch_size = config.get('batch_size', 32)
    
    print(f"\n Loading data...")
    train_loader, dataset_size = load_data(
        data_dir=data_dir,
        labels_csv=labels_csv,
        batch_size=batch_size,
        split=split,
        max_subset_per_class=max_subset_per_class,
        num_workers=data_config.get('num_workers', 0),
        augment=augment,
        device=device
    )
    
    if train_loader is None:
        print(" Failed to load data")
        sys.exit(1)
    
    # Calculate augmentation statistics
    total_epochs = config['epochs']
    samples_per_class = dataset_size // 10  # Assuming 10 digit classes
    print(f"\n   DATASET STATISTICS:")
    print(f"   Base dataset size: {dataset_size:,} samples")
    print(f"   Samples per class: ~{samples_per_class:,}")
    print(f"   Training epochs: {total_epochs}")
    
    if augment:
        total_augmented_samples = dataset_size * total_epochs
        augmented_per_class = samples_per_class * total_epochs
        print(f"      AUGMENTATION ENABLED:")
        print(f"      - Total augmented samples over training: {total_augmented_samples:,}")
        print(f"      - Augmented samples per class: ~{augmented_per_class:,}")
        print(f"      - Augmentation multiplier: {total_epochs}x original dataset")
        print(f"      - Each sample gets different random augmentation each epoch!")
    else:
        print(f"       AUGMENTATION DISABLED:")
        print(f"      - Same {dataset_size:,} images repeated each epoch")
        print(f"      - No variation between epochs")
        print(f"      - Total training samples: {dataset_size:,} (no multiplier)")
    
    
    model_id = arch_config.get('id','unnamed')
    print(f"\n  Creating models with architecture: {model_id}")
    generator, discriminator = create_models_from_config(arch_config, device)
    
    if not args.resume_from:
        print("Applying initial weights...")
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    trainer = GANTrainer(generator, discriminator, config, device)
    latent_dim = arch_config['generator']['latent_dim']
    
    #  Logic to load checkpoint if provided ---
    start_epoch = 0
    if args.resume_from:
        print(f"Attempting to resume training from: {args.resume_from}")
        try:
            # The new load_checkpoint method returns the epoch number
            start_epoch = trainer.load_checkpoint(args.resume_from)
        except Exception as e:
            print(f" Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0 # Ensure we start from 0 on failure

    checkpoint_dir = Path(config.get('checkpoint_dir'))
    image_dir = Path(config.get('image_dir'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    total_epochs = config['epochs']
    save_every = config.get('save_every', 100)
    gen_save_every = config.get('gen_save_every', 10)
    
    print(f"\n Starting training from epoch {start_epoch + 1} to {start_epoch + total_epochs}...")
    print(f"    Full checkpoint every {save_every} epochs")
    print(f"    Generator checkpoint every {gen_save_every} epochs")
    print("="*60)

    for epoch_idx in range(total_epochs):
        current_epoch = start_epoch + epoch_idx + 1
        avg_g_loss, avg_d_loss = trainer.train_epoch(train_loader, latent_dim)
        
        print(f"Epoch [{current_epoch}/{start_epoch + total_epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
        # Determine if this is the final epoch
        is_final_epoch = (epoch_idx + 1) == total_epochs
        
        # Save full checkpoints at regular intervals OR if it's the final epoch
        should_save_checkpoint = (current_epoch % save_every == 0) or is_final_epoch or epoch_idx == 0
        
        # Save generator more frequently
        should_save_generator = (current_epoch % gen_save_every == 0) or is_final_epoch or epoch_idx == 0
        
        if should_save_checkpoint:
            print(f" Saving full checkpoint at epoch {current_epoch}...")
            trainer.save_checkpoint(current_epoch, checkpoint_dir, force_save=is_final_epoch)
            trainer.save_sample_images(current_epoch, image_dir, latent_dim)
        elif should_save_generator:
            # Save only generator if we're not saving full checkpoint
            print(f" Saving generator at epoch {current_epoch}...")
            trainer.save_generator_only(current_epoch, checkpoint_dir, force_save=is_final_epoch)
            trainer.save_sample_images(current_epoch, image_dir, latent_dim)

    # Save final models
    final_gen_path = checkpoint_dir / 'best_generator.pth'
    torch.save(generator.state_dict(), final_gen_path)

    print(f"\n   Final models saved:")
    print(f"   Generator: {final_gen_path}")

    # Final save for discriminator
    final_disc_path = checkpoint_dir / 'best_discriminator.pth'
    torch.save(discriminator.state_dict(), final_disc_path)
    print(f"   Discriminator: {final_disc_path}")
    
    # Plot training curves
    loss_curve_path = checkpoint_dir.parent / f"training_loss_curve_{model_id}_{nas_method}.png"
    plot_training_curves(trainer.G_losses, trainer.D_losses, loss_curve_path)
    
    print("\n" + "="*60)
    print("   TRAINING COMPLETED!")
    print("="*60)
    print(f"   Final Metrics:")
    print(f"   Final Generator Loss: {trainer.G_losses[-1]:.4f}")
    print(f"   Final Discriminator Loss: {trainer.D_losses[-1]:.4f}")
    print(f"   Total Epochs Trained: {total_epochs}")
    print(f"   Full checkpoints saved: Every {save_every} epochs")
    print(f"   Generator checkpoints saved: Every {gen_save_every} epochs")

if __name__ == '__main__':
    main()