"""
Neural Architecture Search Script

Script to run neural architecture search experiments for Conditional GANs.
A simplified interface to search for optimal architectures using different strategies listed below.

Usage:
    python scripts/nas_search.py --config configs/default_nas.yaml  (runs adaptive search as default)
    
    1)Adaptive search::: python scripts/nas_search.py --strategy adaptive --num-archs 4 --epochs 4
    2)Progressive search::: python scripts/nas_search.py --strategy progressive --num-archs 4 --epochs 4
    3)Multifidelity search::: python scripts/nas_search.py --strategy multifidelity --num-archs 4
    4)Random search::: python scripts/nas_search.py --strategy random --num-archs 4
    5)Adversarial (gradient-based) search::: python scripts/nas_search.py --strategy adversarial --epochs 25
"""

import argparse
import json
import sys
import os
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))


try:
    from nepscript.nas.engine import NASEngine
    from nepscript.utils.data import load_data
    from nepscript.utils.config import (
        load_config, save_config, validate_nas_config, 
        get_default_nas_config, merge_configs, resolve_device
    )
    
except ImportError as e:
    print(f" Import Error: {e}")
    print("Make sure you're running from the project root directory")
    print("Current directory:", os.getcwd())
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Neural Architecture Search for Conditional DCGAN'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['adaptive', 'progressive', 'multifidelity', 'random', 'adversarial'],
        help='Search strategy to use'
    )
    
    parser.add_argument(
        '--num-archs',
        type=int,
        help='Number of architectures to evaluate'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Base number of training epochs per architecture'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--labels-csv',
        type=str,
        help='Path to labels CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps', 'auto'],
        default='auto',
        help='Device to use (auto, mps, cuda, or cpu)'
    )
    
    parser.add_argument(
        '--max-subset-per-class',
        type = int,
        help = "Maximum number of samples per class for the dataset subset"
    )
    
    parser.add_argument(
        '--split',
        type = str,
        help = 'Use data from Train or Test or None for both data separated dirs'
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    config_path = args.config if args.config else 'configs/nas_config.yaml'
    print(f'Loading Configuration from : {config_path}')
    
    try: 
        config = load_config(config_path)
    except FileNotFoundError:
        print(f" Error : Configuration file not found at '{config_path}")
        print('Using hardcoded default config from nepscript.utils.config.py')
        config = get_default_nas_config()
    

    # Override with command line arguments
    overrides = {}
    if args.strategy:
        overrides['strategy'] = args.strategy
    if args.num_archs:
        overrides['num_architectures'] = args.num_archs
    if args.epochs:
        overrides['base_epochs'] = args.epochs
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.device:
        overrides['device'] = args.device
    
    if args.data_dir or args.labels_csv:
        overrides['data'] = overrides.get('data', {})
        if args.data_dir:
            overrides['data']['data_dir'] = args.data_dir
        if args.labels_csv:
            overrides['data']['labels_csv'] = args.labels_csv
        #type of overrides with saftey checks implemented 
    if args.max_subset_per_class is not None or args.split is not None:
        overrides.setdefault('data',{})
    if args.max_subset_per_class is not None:
        overrides['data']['max_subset_per_class'] = args.max_subset_per_class if args.max_subset_per_class > 0 else None 
    if args.split is not None:
        overrides['data']['split'] = None if args.split.lower() == 'none' else args.split 
        
    config = merge_configs(config, overrides)
    
    
    # Validate configuration
    try:
        validate_nas_config(config)
    except ValueError as e:
        print(f" !! Configuration error: {e}")
        sys.exit(1)
    
    # Setup device
    device = resolve_device(config.get('device', 'auto'))
    
    print(f"Using device: {device}")
    
    # Load data
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', 'data/DevanagariHandwrittenDigitDataset')
    labels_csv = data_config.get('labels_csv', 'data/hindi_mnist.csv')
    batch_size = data_config.get('batch_size', 32)
    max_subset = data_config.get('max_subset_per_class', None)
    split = data_config.get('split', None)
    
    print(f"\n Loading data from: {data_dir}")
    train_loader, dataset_size = load_data(
        data_dir=data_dir,
        labels_csv=labels_csv,
        batch_size=batch_size,
        max_subset_per_class=max_subset,
        split=split,
        device=device
    )
    
    if train_loader is None:
        print(" Failed to load data")
        sys.exit(1)
    
    # Initialize NAS engine
    print(f"\n Initializing NAS with {config['strategy']} strategy")
    nas_engine = NASEngine(
        strategy=config['strategy'],
        config=config,
        device=device
    )
    
    # Run search
    results, best_config = nas_engine.run_search(
        train_loader=train_loader,
        num_architectures=config['num_architectures'],
        base_epochs=config['base_epochs']
    )
    
    # Save results
    if config.get('save_results', True):
        output_dir = Path(config.get('output_dir', 'experiments/nas_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f'nas_results_{config["strategy"]}.json'
        output_data = {
            'config': config,
            'results': results,
            'best_architecture': best_config,
            'summary': {
                'total_architectures': len(results),
                'successful_architectures': len([r for r in results if 'error' not in r]),
                'best_score': nas_engine.best_score
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n Results saved to: {results_file}")
        
        # Save best config separately
        if best_config:
            best_config_file = output_dir / f'best_architecture_{config["strategy"]}.json'
            with open(best_config_file, 'w') as f:
                json.dump(best_config, f, indent=2)
            print(f" Best architecture saved to: {best_config_file}")
    
    print("\n" + "="*60)
    print(" NAS EXPERIMENT COMPLETED!")
    print("="*60)
    
    if best_config:
        print(f" Best Architecture: {best_config['id']}")
        print(f" Best Score: {nas_engine.best_score:.4f}")
        print(f"\nGenerator Config:")
        for key, value in best_config['generator'].items():
            print(f"  {key}: {value}")
        print(f"\nDiscriminator Config:")
        for key, value in best_config['discriminator'].items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()