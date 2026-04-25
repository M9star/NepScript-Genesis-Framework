"""
Configuration Management

This module provides utilities for loading, validating, and managing
YAML configuration files for NepScript Genesis experiments.
"""

import yaml
from pathlib import Path
from typing import Dict, Any



from datetime import datetime



def get_timestamp():
    """Generate human-readable timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration saved to: {output_path}")


def validate_nas_config(config: Dict[str, Any]) -> bool:
    """
    Validate NAS configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    required_keys = ['strategy', 'num_architectures', 'base_epochs']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate strategy
    valid_strategies = ['adaptive', 'progressive', 'multifidelity', 'random', 'adversarial']  # Added adversarial
    if config['strategy'] not in valid_strategies:
        raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
    
    # Validate numeric values (skip for adversarial which doesn't use num_architectures)
    if config['strategy'] != 'adversarial':
        if config['num_architectures'] <= 0:
            raise ValueError("num_architectures must be positive")
    
    if config['base_epochs'] <= 0:
        raise ValueError("base_epochs must be positive")
    
    return True


def validate_training_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    required_keys = ['epochs', 'g_lr', 'd_lr', 'beta1', 'batch_size']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate numeric ranges
    if config['epochs'] <= 0:
        raise ValueError("epochs must be positive")
    
    if not (0 < config['g_lr'] < 1):
        raise ValueError("g_lr must be between 0 and 1")
    
    if not (0 < config['d_lr'] < 1):
        raise ValueError("d_lr must be between 0 and 1")
    
    if not (0 < config['beta1'] < 1):
        raise ValueError("beta1 must be between 0 and 1")
    
    if config['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base
        
    Returns:
        dict: Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_default_nas_config() -> Dict[str, Any]:
    """Get default NAS configuration"""
    timestamp = get_timestamp()
    return {
        'strategy': 'adaptive',
        'num_architectures': 10,
        'base_epochs': 8,
        'data': {
            'batch_size': 32,
            'max_subset_per_class': 2000,                   #this is being used by resume training 
            'num_workers': 4
        },
        'device': 'cuda',
        'save_results': True,
        'output_dir': f'experiments/gan_run_models_and_images/nas_results/{timestamp}'
    }


def get_default_training_config(nas_method: str = "unknown") -> Dict[str, Any]:
    """Get default training configuration"""
    safe_nas_method = nas_method.replace('_','-')
    timestamp = get_timestamp()
    base_path = f'experiments/gan_run_models_and_images/final_training/{safe_nas_method}/{timestamp}'
    return {
        'epochs': 400,
        'g_lr': 0.0002,
        'd_lr': 0.0002,
        'beta1': 0.5,
        'batch_size': 32,
        'save_every': 50,
        'gen_save_every': 10, 
        'sampling_strategy': safe_nas_method,
        'checkpoint_dir': f'{base_path}/models',
        'image_dir': f'{base_path}/images',
        'device': 'cuda'
    }