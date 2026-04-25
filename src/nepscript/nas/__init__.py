"""
Neural Architecture Search Module

Implements various search strategies for finding optimal
GAN architectures including adaptive, progressive, multi-fidelity,
and gradient-based (AdversarialNAS) approaches.
"""

from .engine import NASEngine
from .strategies import (
    AdaptiveRandomSearch,
    ProgressiveSearch,
    MultiFidelitySearch
)
from .evaluator import (
    evaluate_architecture,
    EnsembleEvaluator
)
from .adversarial_nas import AdversarialNASEngine
from .supernet import DifferentiableGenerator, DifferentiableDiscriminator
from .differentiable_ops import (
    MixedNorm,
    MixedActivation,
    MixedDropout,
    MixedResidual
)

__all__ = [
    'NASEngine',
    'AdaptiveRandomSearch',
    'ProgressiveSearch',
    'MultiFidelitySearch',
    'AdversarialNASEngine',
    'DifferentiableGenerator',
    'DifferentiableDiscriminator',
    'MixedNorm',
    'MixedActivation',
    'MixedDropout',
    'MixedResidual',
    'evaluate_architecture',
    'EnsembleEvaluator'
]