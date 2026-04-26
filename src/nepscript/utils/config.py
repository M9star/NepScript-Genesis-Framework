"""
Configuration Management

This module provides utilities for loading, validating, and managing
YAML configuration files for NepScript Genesis experiments.
"""

import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def get_timestamp() -> str:
    """Return a human-readable timestamp string ``YYYYMMDD_HHMMSS``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_best_available_device() -> str:
    """
    Auto-detect the best available compute device.

    Priority order: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.

    Returns:
        str: One of ``'cuda'``, ``'mps'``, or ``'cpu'``.
    """
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_device(requested: str = 'auto') -> str:
    """
    Resolve and validate a requested device string.

    This is the **single entry-point** for device resolution across the entire
    codebase.  Pass any value that may come from a YAML config or CLI argument.

    - ``'auto'``       — picks the best available device automatically.
    - ``'cuda'``       — uses CUDA; falls back to CPU with a warning if unavailable.
    - ``'mps'``        — uses MPS; falls back to CPU with a warning if unavailable.
    - ``'cpu'``        — always honoured.
    - anything else   — logs a warning and auto-detects.

    Args:
        requested: Device string from config or CLI (default: ``'auto'``).

    Returns:
        str: Validated device string ready for ``torch.device()``.
    """
    requested = (requested or 'auto').strip().lower()

    if requested == 'auto':
        device = get_best_available_device()
        print(f"Device auto-detected: {device}")
        return device

    if requested == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        print("WARNING: CUDA requested but not available — falling back to CPU.")
        return 'cpu'

    if requested == 'mps':
        if torch.backends.mps.is_available():
            return 'mps'
        print("WARNING: MPS requested but not available — falling back to CPU.")
        return 'cpu'

    if requested == 'cpu':
        return 'cpu'

    print(f"WARNING: Unknown device '{requested}' — auto-detecting best device.")
    return get_best_available_device()


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to: {output_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_nas_config(config: Dict[str, Any]) -> bool:
    """
    Validate a NAS configuration dictionary.

    Args:
        config: Configuration dictionary.

    Returns:
        bool: ``True`` if valid.

    Raises:
        ValueError: On any invalid value.
    """
    required_keys = ['strategy', 'num_architectures', 'base_epochs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    valid_strategies = ['adaptive', 'progressive', 'multifidelity', 'random', 'adversarial']
    if config['strategy'] not in valid_strategies:
        raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")

    # Adversarial search derives a single architecture — num_architectures is unused.
    if config['strategy'] != 'adversarial' and config['num_architectures'] <= 0:
        raise ValueError("num_architectures must be positive")

    if config['base_epochs'] <= 0:
        raise ValueError("base_epochs must be positive")

    return True


def validate_training_config(config: Dict[str, Any]) -> bool:
    """
    Validate a GAN training configuration dictionary.

    Args:
        config: Configuration dictionary.

    Returns:
        bool: ``True`` if valid.

    Raises:
        ValueError: On any invalid value.
    """
    required_keys = ['epochs', 'g_lr', 'd_lr', 'beta1', 'batch_size']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

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


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Values in *override_config* take precedence over *base_config*.
    Nested dicts are merged recursively; all other types are replaced.

    Args:
        base_config: Base configuration.
        override_config: Configuration whose values override the base.

    Returns:
        dict: Merged configuration dictionary.
    """
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

def get_default_nas_config() -> Dict[str, Any]:
    """
    Return sensible default NAS configuration.

    Device is set to ``'auto'`` so :func:`resolve_device` picks the best
    available backend at runtime on any machine.
    """
    timestamp = get_timestamp()
    return {
        'strategy': 'adaptive',
        'num_architectures': 10,
        'base_epochs': 8,
        'data': {
            'batch_size': 32,
            'max_subset_per_class': 2000,
            'num_workers': 4,
        },
        'device': 'auto',
        'save_results': True,
        'output_dir': f'experiments/gan_run_models_and_images/nas_results/{timestamp}',
    }


def get_default_training_config(nas_method: str = "unknown") -> Dict[str, Any]:
    """
    Return sensible default GAN training configuration.

    Device is set to ``'auto'`` so :func:`resolve_device` picks the best
    available backend at runtime on any machine.

    Args:
        nas_method: NAS strategy name — used to construct output directory paths.
    """
    safe_nas_method = nas_method.replace('_', '-')
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
        'device': 'auto',
    }