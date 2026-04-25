"""
Differentiable Operations for AdversarialNAS

This module implements differentiable operations that mix multiple choices
using learnable architecture parameters (α). These operations enable
gradient-based neural architecture search for GANs.

Key Concept:
    Instead of picking ONE operation (e.g., BatchNorm OR InstanceNorm),
    use a weighted combination controlled by learnable parameters α:
    
    output = softmax(α)[0] * BatchNorm(x) + softmax(α)[1] * InstanceNorm(x)
    
    The α parameters are optimized via gradient descent to find the best
    architectural choices.

Reference:
    Gong et al. "AdversarialNAS: Adversarial Neural Architecture Search for GANs"
    CVPR 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.blocks import GeneratorResBlock, DiscriminatorResBlock


class MixedNorm(nn.Module):
    """
    Differentiable normalization mixing BatchNorm and InstanceNorm
    
    During search:
        output = softmax(α)[0] * BatchNorm(x) + softmax(α)[1] * InstanceNorm(x)
    
    After search:
        Derive discrete choice based on argmax(α)
    
    Args:
        channels: Number of channels for normalization layers
    """
    
    def __init__(self, channels):
        super().__init__()
        
        # The two normalization operations
        self.ops = nn.ModuleList([
            nn.BatchNorm2d(channels),
            nn.InstanceNorm2d(channels)
        ])
        
        # Architecture parameter (learnable!)
        # Initialized randomly, will be optimized during search
        self.α = nn.Parameter(torch.randn(2))
        
        self.op_names = ['batch', 'instance']
    
    def forward(self, x):
        """
        Forward pass using weighted combination
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Mixed normalization output
        """
        # Compute softmax weights (ensures sum to 1 and all positive)
        weights = F.softmax(self.α, dim=0)
        
        # Weighted sum of operations
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        
        return output
    
    def derive_discrete_choice(self):
        """
        Extract the dominant normalization choice after search
        
        Returns:
            str: 'batch' or 'instance'
        """
        return self.op_names[self.α.argmax().item()]
    
    def get_weights(self):
        """Get current softmax weights for analysis"""
        with torch.no_grad():
            return F.softmax(self.α, dim=0).cpu().numpy()


class MixedActivation(nn.Module):
    """
    Differentiable activation mixing ReLU and LeakyReLU
    
    Search Space: {ReLU, LeakyReLU(0.2)}
    
    Args:
        None
    """
    
    def __init__(self):
        super().__init__()
        
        # The two activation functions
        self.ops = nn.ModuleList([
            nn.ReLU(inplace=False),  # Can't use inplace with mixing
            nn.LeakyReLU(0.2, inplace=False)
        ])
        
        # Architecture parameter
        self.α = nn.Parameter(torch.randn(2))
        
        self.op_names = ['relu', 'leaky_relu']
    
    def forward(self, x):
        """
        Forward pass using weighted combination
        
        Args:
            x: Input tensor
            
        Returns:
            Mixed activation output
        """
        weights = F.softmax(self.α, dim=0)
        
        # Apply each activation and mix
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        
        return output
    
    def derive_discrete_choice(self):
        """
        Extract the dominant activation after search
        
        Returns:
            str: 'relu' or 'leaky_relu'
        """
        return self.op_names[self.α.argmax().item()]
    
    def get_weights(self):
        """Get current softmax weights for analysis"""
        with torch.no_grad():
            return F.softmax(self.α, dim=0).cpu().numpy()


class MixedDropout(nn.Module):
    """
    Differentiable dropout rate selection
    
    Search Space: {0.0, 0.1, 0.2, 0.3}
    
    Special handling needed because dropout is stochastic
    
    Args:
        None
    """
    
    def __init__(self):
        super().__init__()
        
        # Discrete dropout rate choices
        self.rates = [0.0, 0.1, 0.2, 0.3]
        
        # Architecture parameter
        self.α = nn.Parameter(torch.randn(len(self.rates)))
    
    def forward(self, x):
        """
        Forward pass using weighted combination of dropout rates
        
        Args:
            x: Input tensor
            
        Returns:
            Mixed dropout output
        """
        weights = F.softmax(self.α, dim=0)
        
        # Apply each dropout rate and mix
        output = 0
        for w, rate in zip(weights, self.rates):
            if rate > 0:
                # Apply dropout with this rate
                output += w * F.dropout(x, p=rate, training=self.training)
            else:
                # No dropout
                output += w * x
        
        return output
    
    def derive_discrete_choice(self):
        """
        Extract the dominant dropout rate after search
        
        Returns:
            float: Chosen dropout rate
        """
        return self.rates[self.α.argmax().item()]
    
    def get_weights(self):
        """Get current softmax weights for analysis"""
        with torch.no_grad():
            return F.softmax(self.α, dim=0).cpu().numpy()


class MixedResidual(nn.Module):
    """
    Differentiable residual connection choice
    
    Options:
        - Identity (skip): output = x
        - ResBlock: output = ResBlock(x)
    
    Args:
        channels: Number of channels for ResBlock
        norm_type: Normalization type for ResBlock ('batch' or 'instance')
        activation: Activation type for ResBlock ('relu' or 'leaky_relu')
        block_type: 'generator' or 'discriminator' (determines which ResBlock to use)
    """
    
    def __init__(self, channels, norm_type='batch', activation='relu', block_type='generator'):
        super().__init__()
        
        # Create the two operations
        if block_type == 'generator':
            res_block = GeneratorResBlock(
                channels=channels,
                norm_type=norm_type,
                activation=activation
            )
        else:  # discriminator
            res_block = DiscriminatorResBlock(
                channels=channels,
                norm_type=norm_type,
                activation=activation
            )
        
        self.ops = nn.ModuleList([
            nn.Identity(),  # Skip connection (no residual)
            res_block       # Use residual block
        ])
        
        # Architecture parameter
        self.α = nn.Parameter(torch.randn(2))
        
        self.op_names = [False, True]  # use_residual boolean
    
    def forward(self, x):
        """
        Forward pass using weighted combination
        
        Args:
            x: Input tensor
            
        Returns:
            Mixed residual output
        """
        weights = F.softmax(self.α, dim=0)
        
        # Mix identity and residual
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        
        return output
    
    def derive_discrete_choice(self):
        """
        Extract whether to use residual connection
        
        Returns:
            bool: True if using residual, False otherwise
        """
        return self.op_names[self.α.argmax().item()]
    
    def get_weights(self):
        """Get current softmax weights for analysis"""
        with torch.no_grad():
            return F.softmax(self.α, dim=0).cpu().numpy()


class MixedOperation(nn.Module):
    """
    Generic mixed operation wrapper (for potential extensions)
    
    This can be used to create custom mixed operations beyond the
    predefined ones above.
    
    Args:
        operations: List of nn.Module operations to mix
        operation_names: List of names for each operation
    """
    
    def __init__(self, operations, operation_names):
        super().__init__()
        
        assert len(operations) == len(operation_names), \
            "Number of operations must match number of names"
        
        self.ops = nn.ModuleList(operations)
        self.op_names = operation_names
        
        # Architecture parameter
        self.α = nn.Parameter(torch.randn(len(operations)))
    
    def forward(self, x):
        """Forward pass using weighted combination"""
        weights = F.softmax(self.α, dim=0)
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        return output
    
    def derive_discrete_choice(self):
        """Extract the dominant operation"""
        return self.op_names[self.α.argmax().item()]
    
    def get_weights(self):
        """Get current softmax weights"""
        with torch.no_grad():
            return F.softmax(self.α, dim=0).cpu().numpy()


def collect_architecture_parameters(model):
    """
    Collect all architecture parameters (α) from a model
    
    This helper function traverses the model and collects all α parameters
    from mixed operations for architecture optimization.
    
    Args:
        model: nn.Module containing mixed operations
        
    Returns:
        list: All α parameters
    """
    arch_params = []
    
    for module in model.modules():
        if isinstance(module, (MixedNorm, MixedActivation, MixedDropout, 
                              MixedResidual, MixedOperation)):
            arch_params.append(module.α)
    
    return arch_params


def derive_architecture_config(model):
    """
    Derive discrete architecture configuration from a model with mixed operations
    
    Args:
        model: nn.Module containing mixed operations
        
    Returns:
        dict: Architecture configuration with discrete choices
    """
    config = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (MixedNorm, MixedActivation, MixedDropout, 
                              MixedResidual, MixedOperation)):
            choice = module.derive_discrete_choice()
            config[name] = choice
    
    return config


def get_architecture_weights(model):
    """
    Get current softmax weights for all mixed operations (for visualization)
    
    Args:
        model: nn.Module containing mixed operations
        
    Returns:
        dict: Mapping from operation name to current weights
    """
    weights = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (MixedNorm, MixedActivation, MixedDropout, 
                              MixedResidual, MixedOperation)):
            weights[name] = module.get_weights()
    
    return weights
