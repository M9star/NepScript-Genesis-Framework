"""
Model Evaluation Script - FIXED VERSION

Fixes:
1. Save results with architecture ID and sampling strategy in filename
2. Better strategy extraction from model paths (handles adaptive-mutation, progressive-phase-3, etc.)
3. Show architecture ID and sampling strategy in visualization
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from datetime import datetime
import re

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from nepscript.models.factory import create_models_from_config, calculate_model_size
from nepscript.nas.evaluator import evaluate_architecture, EnsembleEvaluator
from nepscript.utils.config import resolve_device


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained GAN models using NAS evaluation metrics'
    )
    
    parser.add_argument('--generator', type=str, help='Path to trained generator model (.pth file)')
    parser.add_argument('--discriminator', type=str, help='Path to trained discriminator model (.pth file)')
    parser.add_argument('--arch-config', type=str, help='Path to architecture configuration JSON file')
    parser.add_argument('--eval-dir', type=str, help='Directory containing models to evaluate (batch mode)')
    parser.add_argument('--recursive', action='store_true', help='Recursively search for models in subdirectories')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file (contains both G and D)')
    parser.add_argument('--output-dir', type=str, default='experiments/evaluation', help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps', 'auto'], default='auto', help='Device to use (auto, mps, cuda, or cpu)')
    parser.add_argument('--use-enhanced', action='store_true', default=True, help='Use enhanced ensemble evaluation (default: True)')
    
    return parser.parse_args()


def parse_model_path_info(model_path: Path):
    """
    Parses a model's file path to extract its strategy, timestamp, and base name.
    This is robust to paths with or without a timestamp directory.
    """
    info = {
        "strategy": "unknown",
        "timestamp": None,
        "model_name": model_path.stem
    }

    try:
        parts = model_path.parts
        
        # Find the base strategy folder, which is right after 'final_training'
        if 'final_training' in parts:
            training_idx = parts.index('final_training')
            
            if training_idx + 1 < len(parts):
                strategy_folder = parts[training_idx + 1]
                
                # Check for timestamp directory between strategy and 'models'
                if training_idx + 2 < len(parts) and parts[training_idx + 2] != 'models':
                    potential_timestamp = parts[training_idx + 2]
                    # Regex for 'YYYYMMDD_HHMMSS' or 'YYYY-MM-DD HH:MM:SS'
                    if (re.match(r'\d{8}_\d{6}', potential_timestamp) or 
                        re.match(r'\d{4}-\d{2}-\d{2}', potential_timestamp)):
                        info['timestamp'] = potential_timestamp

                # Extract the base strategy name (e.g., 'progressive-phase-2' -> 'progressive')
                if 'adaptive' in strategy_folder.lower():
                    info['strategy'] = 'adaptive'
                elif 'progressive' in strategy_folder.lower():
                    info['strategy'] = 'progressive'
                elif 'multifidelity' in strategy_folder.lower():
                    info['strategy'] = 'multifidelity'
                elif 'random' in strategy_folder.lower():
                    info['strategy'] = 'random'
                else:
                    info['strategy'] = strategy_folder # Fallback
                    
    except (ValueError, IndexError):
        # Fallback if path is not in the expected format, use the old method
        info['strategy'] = extract_strategy_from_path(model_path)
        
    return info



def find_discriminator_for_generator(gen_path):
    """
    Find the corresponding discriminator model for a generator.
    
    Tries multiple strategies:
    1. Same directory, replace 'generator' with 'discriminator'
    2. Same directory, look for 'best_discriminator.pth'
    3. Parent directory pattern matching
    """
    gen_path = Path(gen_path)
    
    # Strategy 1: Direct replacement
    disc_path = gen_path.parent / gen_path.name.replace('generator', 'discriminator')
    if disc_path.exists():
        return str(disc_path)
    
    # Strategy 2: Look for best_discriminator.pth in same directory
    disc_path = gen_path.parent / 'best_discriminator.pth'
    if disc_path.exists():
        return str(disc_path)
    
    # Strategy 3: Look for any discriminator file with matching epoch
    epoch_match = re.search(r'epoch_(\d+)', gen_path.name)
    if epoch_match:
        epoch = epoch_match.group(1)
        disc_pattern = f"*discriminator*epoch_{epoch}*.pth"
        disc_files = list(gen_path.parent.glob(disc_pattern))
        if disc_files:
            return str(disc_files[0])
    
    # Strategy 4: Look for checkpoint file
    if 'best' in gen_path.name:
        checkpoint_path = gen_path.parent / 'checkpoint_final.pth'
        if checkpoint_path.exists():
            return 'from_checkpoint'
    
    return None


def extract_strategy_from_path(model_path):
    """
    Extract base strategy name from model path.
    
    Handles variations like:
        - adaptive-mutation -> adaptive
        - progressive-phase-3 -> progressive
        - multifidelity -> multifidelity
        - random -> random
    """
    path = Path(model_path)
    parts = path.parts
    
    # Look for known strategy patterns in path
    for part in reversed(parts):
        part_lower = part.lower()
        
        # Check for base strategies
        if 'adaptive' in part_lower:
            return 'adaptive'
        elif 'progressive' in part_lower:
            return 'progressive'
        elif 'multifidelity' in part_lower:
            return 'multifidelity'
        elif 'random' in part_lower:
            return 'random'
    
    return 'unknown'


def find_config_for_strategy(strategy):
    """Find architecture config file for a given strategy (base name only)."""
    nas_results_dir = Path("experiments/gan_run_models_and_images/nas_results")
    
    # Map strategy to expected config file
    config_path = nas_results_dir / f"best_architecture_{strategy}.json"
    if config_path.exists():
        return str(config_path)
    
    # Fallback: look for any matching config
    pattern = f"*{strategy}*.json"
    matches = list(nas_results_dir.glob(pattern))
    if matches:
        # Prefer best_architecture files
        best_matches = [m for m in matches if 'best_architecture' in m.name]
        if best_matches:
            return str(best_matches[0])
        return str(matches[0])
    
    return None


def get_architecture_id_and_strategy(arch_config):
    """
    Extract architecture ID and sampling strategy from config.
    
    Args:
        arch_config: Architecture configuration dictionary
        
    Returns:
        tuple: (architecture_id, sampling_strategy)
    """
    arch_id = arch_config.get('id', 'unknown')
    
    # Try to get sampling_strategy from config
    sampling_strategy = arch_config.get('sampling_strategy')
    
    # If not found, try to infer from ID or use 'random' as default
    if not sampling_strategy:
        # Check if strategy is mentioned in the ID
        if 'adaptive' in arch_id.lower():
            sampling_strategy = 'adaptive'
        elif 'progressive' in arch_id.lower():
            sampling_strategy = 'progressive'
        elif 'multifidelity' in arch_id.lower():
            sampling_strategy = 'multifidelity'
        else:
            # Default to 'random' if no strategy found
            sampling_strategy = 'random'
    
    return arch_id, sampling_strategy


def reconstruct_training_stats(generator, discriminator, device='auto'):
    """Reconstruct minimal training statistics needed for evaluation."""
    return {
        'stable_epochs': 10,
        'diverged': False,
        'final_g_loss': 1.0,
        'final_d_loss': 1.0,
        'training_converged': True
    }


def evaluate_single_model(gen_path, disc_path, arch_config, device='auto', use_enhanced=True):
    """Evaluate a single generator-discriminator pair."""
    print(f"\n{'='*60}")
    print(f"Evaluating Model: {Path(gen_path).stem}")
    print(f"{'='*60}")
    
    try:
        # Auto-detect discriminator if not provided
        if not disc_path or disc_path == 'auto':
            print(" Auto-detecting discriminator...")
            disc_path = find_discriminator_for_generator(gen_path)
            if not disc_path:
                return {
                    'error': 'Could not find corresponding discriminator model',
                    'generator_path': str(gen_path)
                }
            print(f" Found discriminator: {Path(disc_path).name}")
        
        # Extract architecture ID and sampling strategy
        arch_id, sampling_strategy = get_architecture_id_and_strategy(arch_config)
        print(f" Architecture ID: {arch_id}")
        print(f" Sampling Strategy: {sampling_strategy}")
        
        # Create models
        generator, discriminator = create_models_from_config(arch_config, device)
        
        # Load weights
        print(" Loading model weights...")
        generator.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True))
        discriminator.load_state_dict(torch.load(disc_path, map_location=device, weights_only=True))
        
        generator.eval()
        discriminator.eval()
        
        # Get model sizes
        gen_size = calculate_model_size(generator)
        disc_size = calculate_model_size(discriminator)
        
        print(f"✓ Models loaded - G: {gen_size:.2f}M params, D: {disc_size:.2f}M params")
        
        # Reconstruct training stats
        training_stats = reconstruct_training_stats(generator, discriminator, device)
        
        # Evaluate
        print(" Running evaluation...")
        if use_enhanced:
            evaluator = EnsembleEvaluator()
            results = evaluator.evaluate_architecture_advanced(
                generator, discriminator, arch_config, training_stats
            )
        else:
            results = evaluate_architecture(
                generator, discriminator, arch_config, training_stats
            )
        
        # Add model info with architecture ID and sampling strategy
        results['model_info'] = {
            'generator_path': str(gen_path),
            'discriminator_path': str(disc_path),
            'generator_size_mb': gen_size,
            'discriminator_size_mb': disc_size,
            'total_size_mb': gen_size + disc_size,
            'architecture_id': arch_id,
            'sampling_strategy': sampling_strategy
        }
        
        # Print results
        print(f"\n Evaluation Results:")
        if use_enhanced:
            print(f"   Enhanced Score: {results.get('enhanced_score', 0):.4f}")
            if 'score_breakdown' in results:
                print(f"   Score Breakdown:")
                for key, value in results['score_breakdown'].items():
                    print(f"    - {key}: {value:.4f}")
        else:
            print(f"   Overall Score: {results.get('overall_score', 0):.4f}")
            print(f"   Stability: {results.get('stability_score', 0):.4f}")
            print(f"    Loss Balance: {results.get('loss_balance_score', 0):.4f}")
            print(f"   Generation Quality: {results.get('generation_quality_score', 0):.4f}")
        
        return results
        
    except Exception as e:
        print(f" Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'generator_path': str(gen_path),
            'discriminator_path': str(disc_path) if disc_path else 'unknown'
        }


def find_model_pairs(eval_dir, recursive=False):
    """Find all generator-discriminator pairs in a directory."""
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        print(f" Directory not found: {eval_dir}")
        return []
    
    pairs = []
    
    if recursive:
        # Look in strategy subdirectories
        for strategy_dir in eval_path.iterdir():
            if strategy_dir.is_dir():
                models_dir = strategy_dir / "models"
                if models_dir.exists():
                    gen_files = list(models_dir.glob("best_generator.pth"))
                    disc_files = list(models_dir.glob("best_discriminator.pth"))
                    
                    if gen_files and disc_files:
                        # Extract base strategy name
                        base_strategy = extract_strategy_from_path(strategy_dir)
                        pairs.append((gen_files[0], disc_files[0], base_strategy))
    else:
        # Look in current directory
        gen_files = list(eval_path.glob("best_generator.pth"))
        disc_files = list(eval_path.glob("best_discriminator.pth"))
        
        if gen_files and disc_files:
            base_strategy = extract_strategy_from_path(eval_path)
            pairs.append((gen_files[0], disc_files[0], base_strategy))
    
    return pairs


def batch_evaluate(eval_dir, recursive=False, device='auto', use_enhanced=True, output_dir='experiments/evaluation'):
    """Batch evaluate all models in a directory."""
    print(f"\n Searching for models in: {eval_dir}")
    print(f"   Recursive: {recursive}")
    
    pairs = find_model_pairs(eval_dir, recursive)
    
    if not pairs:
        print(" No model pairs found!")
        return None
    
    print(f"✓ Found {len(pairs)} model pair(s) to evaluate")
    
    results = []
    
    for gen_path, disc_path, base_strategy in pairs:
        print(f"\n{'='*60}")
        print(f"Base Strategy: {base_strategy}")
        print(f"{'='*60}")
        
        # Try to find config for this base strategy
        config_path = find_config_for_strategy(base_strategy)
        
        if not config_path:
            print(f"  Config not found for strategy: {base_strategy}")
            print("   Skipping this model...")
            continue
        
        print(f" Using config: {Path(config_path).name}")
        
        with open(config_path, 'r') as f:
            arch_config = json.load(f)
        
        result = evaluate_single_model(
            gen_path, disc_path, arch_config, device, use_enhanced
        )
        
        if result and 'error' not in result:
            result['strategy'] = base_strategy
            results.append(result)
        elif 'error' in result:
            print(f"  Evaluation failed: {result['error']}")
    
    if results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"batch_evaluation_{timestamp}.json"
        
        batch_data = {
            'timestamp': timestamp,
            'evaluation_dir': str(eval_dir),
            'num_models': len(results),
            'device': device,
            'enhanced_evaluation': use_enhanced,
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(batch_data, f, indent=2, default=str)
        
        print(f"\n Batch evaluation completed!")
        print(f" Results saved to: {results_file}")
        
        print(f"\n Summary:")
        for result in results:
            strategy = result.get('strategy', 'unknown')
            score = result.get('enhanced_score', result.get('overall_score', 0))
            arch_id = result.get('model_info', {}).get('architecture_id', 'unknown')
            print(f"  {strategy} ({arch_id}): {score:.4f}")
        
        return batch_data
    
    return None



def main():
    """Main execution function"""
    args = parse_args()
    
    # Setup device
    device = resolve_device(args.device)
    
    print(f"  Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine evaluation mode
    if args.eval_dir:
        # Batch evaluation mode remains unchanged as it saves a single file
        batch_evaluate(
            args.eval_dir,
            recursive=args.recursive,
            device=device,
            use_enhanced=args.use_enhanced,
            output_dir=args.output_dir
        )
    
    elif args.generator:
        # Single model evaluation
        gen_path = Path(args.generator)
        disc_path_str = args.discriminator
        
        # --- MODIFIED SECTION ---
        # Use the new robust parser to get strategy and timestamp
        path_info = parse_model_path_info(gen_path)
        
        # Auto-detect discriminator if not provided
        if not disc_path_str:
            print("  No discriminator specified, attempting auto-detection...")
            disc_path_str = find_discriminator_for_generator(gen_path)
            if not disc_path_str:
                print("  Could not find discriminator model!")
                print("   Please specify --discriminator explicitly")
                sys.exit(1)
        
        # Try to find arch config using the robustly parsed strategy
        if not args.arch_config:
            print("  No architecture config specified, attempting auto-detection...")
            base_strategy = path_info['strategy']
            config_path = find_config_for_strategy(base_strategy)
            if not config_path:
                print(f"  Could not find architecture config for strategy: {base_strategy}")
                print("   Please specify --arch-config explicitly")
                sys.exit(1)
            print(f" Found config for base strategy '{base_strategy}': {Path(config_path).name}")
        else:
            config_path = args.arch_config
        
        with open(config_path, 'r') as f:
            arch_config = json.load(f)
        
        result = evaluate_single_model(
            str(gen_path),
            disc_path_str,
            arch_config,
            device,
            args.use_enhanced
        )
        
        if result and 'error' not in result:
            arch_id = result['model_info']['architecture_id']
            sampling_strategy = result['model_info']['sampling_strategy']
            
            # --- NEW FILENAME LOGIC ---
            timestamp_suffix = f"_{path_info['timestamp']}" if path_info['timestamp'] else ""
            
            model_name = path_info['model_name']
            # Save results
            results_file = output_dir / f"eval_{model_name}_{arch_id}_{sampling_strategy}{timestamp_suffix}.json"
            
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\n Evaluation completed!")
            print(f"  Results saved to: {results_file}")
        else:
            print(f"\n  Evaluation failed!")
            if result and 'error' in result:
                print(f"   Error: {result['error']}")
            sys.exit(1)
            
    else:
        print("  Error: Must specify either:")
        print("   - --eval-dir for batch evaluation")
        print("   - --generator (--discriminator and --arch-config optional, will auto-detect)")
        sys.exit(1)


if __name__ == '__main__':
    main()