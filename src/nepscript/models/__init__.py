"""
GAN Model Definitions

Contains generator, discriminator, and factory utilities
for creating conditional DCGAN architectures.
"""

from .generator import SearchableGenerator, GENERATOR_SEARCH_SPACE, DeconvBlock
from .discriminator import SearchableDiscriminator, DISCRIMINATOR_SEARCH_SPACE, ConvBlock
from .factory import (
    create_models_from_config,
    weights_init,
    sample_architecture_pair,
    sample_generator_config,
    sample_discriminator_config,
    calculate_model_size,
    get_model_info
)

__all__ = [
    'SearchableGenerator',
    'SearchableDiscriminator',
    'GENERATOR_SEARCH_SPACE',
    'DISCRIMINATOR_SEARCH_SPACE',
    'DeconvBlock',
    'ConvBlock',
    'create_models_from_config',
    'weights_init',
    'sample_architecture_pair',
    'sample_generator_config',
    'sample_discriminator_config',
    'calculate_model_size',
    'get_model_info'
]
