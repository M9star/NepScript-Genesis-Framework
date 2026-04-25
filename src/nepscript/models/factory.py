"""
Model Factory - Utilities for Creating and Managing Models

This module provides utility functions for creating model instances,
initializing weights, and managing model configurations.
"""

import random
import torch
import torch.nn as nn
from .generator import SearchableGenerator, GENERATOR_SEARCH_SPACE
from .discriminator import SearchableDiscriminator, DISCRIMINATOR_SEARCH_SPACE


def weights_init(m):
    """
    Applies custom weights initialization to models.
    This is the robust version that checks for specific layer types.
    
    Args:
        m: PyTorch module to initialize
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def sample_generator_config():
    """Sample random generator hyperparameters (architecture FIXED)"""
    config = {}
    for param, options in GENERATOR_SEARCH_SPACE.items():
        config[param] = random.choice(options)
    return config


def sample_discriminator_config():
    """Sample random discriminator hyperparameters (architecture FIXED)"""
    config = {}
    for param, options in DISCRIMINATOR_SEARCH_SPACE.items():
        config[param] = random.choice(options)
    return config


def sample_architecture_pair():
    """Sample a compatible generator-discriminator architecture pair"""
    gen_config = sample_generator_config()
    disc_config = sample_discriminator_config()
    
    # Ensure compatibility between generator and discriminator
    gen_config['num_classes'] = 10  # Nepali digits 0-9
    disc_config['num_classes'] = 10
    
    # Create a unique architecture ID for tracking
    arch_id = f"G{hash(str(gen_config)) % 10000:04d}_D{hash(str(disc_config)) % 10000:04d}"
    
    return {
        'id': arch_id,
        'generator': gen_config,
        'discriminator': disc_config
    }


def create_models_from_config(config, device='cuda'):
    """
    Create generator and discriminator models from configuration
    
    Args:
        config: Configuration dictionary with 'generator' and 'discriminator' keys
        device: Device to place models on
        
    Returns:
        tuple: (generator, discriminator) models
    """
    generator = SearchableGenerator(config['generator']).to(device)
    discriminator = SearchableDiscriminator(config['discriminator']).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    return generator, discriminator


def calculate_model_size(model):
    """
    Calculate model size in millions of parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Number of parameters in millions
    """
    return sum(p.numel() for p in model.parameters()) / 1e6


def get_model_info(generator, discriminator):
    """
    Get comprehensive information about model sizes
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        
    Returns:
        dict: Dictionary with model size information
    """
    gen_size = calculate_model_size(generator)
    disc_size = calculate_model_size(discriminator)
    total_size = gen_size + disc_size
    
    return {
        'generator_params_m': gen_size,
        'discriminator_params_m': disc_size,
        'total_params_m': total_size
    }