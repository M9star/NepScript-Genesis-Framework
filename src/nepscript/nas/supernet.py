"""
Supernet Architecture for AdversarialNAS

This module implements the differentiable supernet that contains ALL
possible architectural choices. The supernet uses mixed operations
controlled by learnable architecture parameters (α).

Key Concepts:
    - Supernet: Single network containing all possible architectures
    - During search: Uses weighted combinations of operations
    - After search: Derives discrete architecture from learned α values

Reference:
    Gong et al. "AdversarialNAS: Adversarial Neural Architecture Search for GANs"
    CVPR 2019
"""

import torch
import torch.nn as nn
from .differentiable_ops import (
    MixedNorm, MixedActivation, MixedDropout, MixedResidual,
    collect_architecture_parameters
)


class DifferentiableGenerator(nn.Module):
    """
    Generator supernet containing ALL architectural choices
    
    This extends SearchableGenerator by replacing discrete operations
    with differentiable mixed operations.
    
    Architecture (FIXED DCGAN structure):
        - Label embedding
        - Initial projection: latent_dim → 512 × 4 × 4
        - Deconv layers: 512 → 256 → 128
        - Mixed operations at each layer (SEARCHABLE)
        - Final output: 1 × 32 × 32
    
    Searchable Components:
        - Normalization: BatchNorm vs InstanceNorm
        - Activation: ReLU vs LeakyReLU
        - Dropout rate: 0.0, 0.1, 0.2, 0.3
        - Residual connections: True vs False
    
    Args:
        latent_dim: Dimension of input noise vector
        num_classes: Number of classes (10 for Devanagari digits)
    """
    
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # FIXED DCGAN architecture - proven structure
        self.channels = [512, 256, 128]
        
        # Label embedding (conditional GAN)
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Initial projection
        combined_dim = latent_dim + latent_dim  # noise + label embedding
        self.initial = nn.Linear(combined_dim, self.channels[0] * 4 * 4)
        
        # DIFFERENTIABLE layers with mixed operations
        self.layers = nn.ModuleList()
        
        for i in range(len(self.channels) - 1):
            layer_dict = nn.ModuleDict({
                # Fixed deconvolution (DCGAN standard)
                'deconv': nn.ConvTranspose2d(
                    self.channels[i], 
                    self.channels[i + 1],
                    kernel_size=4, 
                    stride=2, 
                    padding=1, 
                    bias=False
                ),
                
                # SEARCHABLE: Normalization type
                'norm': MixedNorm(self.channels[i + 1]),
                
                # SEARCHABLE: Activation function
                'activation': MixedActivation(),
                
                # SEARCHABLE: Dropout rate
                'dropout': MixedDropout(),
                
                # SEARCHABLE: Residual connection
                'residual': MixedResidual(
                    channels=self.channels[i + 1],
                    norm_type='batch',  # Default for ResBlock
                    activation='relu',
                    block_type='generator'
                )
            })
            
            self.layers.append(layer_dict)
        
        # Final output layer (FIXED)
        self.final = nn.ConvTranspose2d(
            self.channels[-1], 
            1,  # Grayscale output
            kernel_size=4, 
            stride=2, 
            padding=1, 
            bias=False
        )
        self.tanh = nn.Tanh()
    
    def forward(self, noise, labels):
        """
        Forward pass through the differentiable generator
        
        Args:
            noise: Random noise tensor (B, latent_dim)
            labels: Class labels (B,)
            
        Returns:
            Generated images (B, 1, 32, 32)
        """
        # Combine noise and label embedding
        label_embed = self.label_embedding(labels)
        x = torch.cat([noise, label_embed], dim=1)
        
        # Initial projection
        x = self.initial(x)
        x = x.view(-1, self.channels[0], 4, 4)
        
        # Pass through differentiable layers
        for layer in self.layers:
            # Deconvolution (fixed)
            x = layer['deconv'](x)
            
            # Mixed normalization (searchable)
            x = layer['norm'](x)
            
            # Mixed activation (searchable)
            x = layer['activation'](x)
            
            # Mixed dropout (searchable)
            x = layer['dropout'](x)
            
            # Mixed residual (searchable)
            x = layer['residual'](x)
        
        # Final output
        x = self.final(x)
        x = self.tanh(x)
        
        return x
    
    def arch_parameters(self):
        """
        Return all architecture parameters (α) for optimization
        
        Returns:
            list: All α parameters from mixed operations
        """
        return collect_architecture_parameters(self)
    
    def derive_architecture(self):
        """
        Extract discrete architecture configuration from learned α values
        
        After search, this method converts the continuous α parameters
        to discrete architectural choices by taking argmax.
        
        Returns:
            dict: Discrete architecture configuration
        """
        # Use the first layer's choices (could aggregate across layers)
        first_layer = self.layers[0]
        
        config = {
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes,
            'norm_type': first_layer['norm'].derive_discrete_choice(),
            'activation': first_layer['activation'].derive_discrete_choice(),
            'dropout_rate': first_layer['dropout'].derive_discrete_choice(),
            'use_residual': first_layer['residual'].derive_discrete_choice()
        }
        
        return config
    
    def get_architecture_weights(self):
        """
        Get current softmax weights for all mixed operations
        
        Useful for monitoring architecture search progress
        
        Returns:
            dict: Current weights for each mixed operation
        """
        weights = {}
        
        for i, layer in enumerate(self.layers):
            weights[f'layer_{i}_norm'] = layer['norm'].get_weights()
            weights[f'layer_{i}_activation'] = layer['activation'].get_weights()
            weights[f'layer_{i}_dropout'] = layer['dropout'].get_weights()
            weights[f'layer_{i}_residual'] = layer['residual'].get_weights()
        
        return weights


class DifferentiableDiscriminator(nn.Module):
    """
    Discriminator supernet containing ALL architectural choices
    
    This extends SearchableDiscriminator by replacing discrete operations
    with differentiable mixed operations.
    
    Architecture (FIXED DCGAN structure):
        - Input: 1 × 32 × 32 (grayscale)
        - Conv layers: 64 → 128 → 256
        - Mixed operations at each layer (SEARCHABLE)
        - Final classification: Real/Fake + Class prediction
    
    Searchable Components:
        - Normalization: BatchNorm vs InstanceNorm
        - Activation: ReLU vs LeakyReLU
        - Dropout rate: 0.0, 0.1, 0.2, 0.3
        - Residual connections: True vs False
    
    Args:
        num_classes: Number of classes (10 for Devanagari digits)
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.num_classes = num_classes
        
        # FIXED DCGAN architecture - proven structure
        self.channels = [64, 128, 256]
        
        # Label embedding (conditional GAN)
        self.label_embedding = nn.Embedding(num_classes, 1 * 32 * 32)
        
        # DIFFERENTIABLE layers with mixed operations
        self.layers = nn.ModuleList()
        
        # First layer (input)
        first_layer = nn.ModuleDict({
            'conv': nn.Conv2d(
                2,  # 1 (image) + 1 (label embedding)
                self.channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            'activation': MixedActivation(),  # No norm on first layer
            'dropout': MixedDropout(),
            'residual': MixedResidual(
                channels=self.channels[0],
                norm_type='batch',
                activation='leaky_relu',
                block_type='discriminator'
            )
        })
        self.layers.append(first_layer)
        
        # Middle layers
        for i in range(len(self.channels) - 1):
            layer_dict = nn.ModuleDict({
                # Fixed convolution (DCGAN standard)
                'conv': nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                
                # SEARCHABLE: Normalization type
                'norm': MixedNorm(self.channels[i + 1]),
                
                # SEARCHABLE: Activation function
                'activation': MixedActivation(),
                
                # SEARCHABLE: Dropout rate
                'dropout': MixedDropout(),
                
                # SEARCHABLE: Residual connection
                'residual': MixedResidual(
                    channels=self.channels[i + 1],
                    norm_type='batch',
                    activation='leaky_relu',
                    block_type='discriminator'
                )
            })
            
            self.layers.append(layer_dict)
        
        # Final classification layer (FIXED)
        self.final = nn.Conv2d(
            self.channels[-1],
            1,  # Real/Fake prediction
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images, labels):
        """
        Forward pass through the differentiable discriminator
        
        Args:
            images: Input images (B, 1, 32, 32)
            labels: Class labels (B,)
            
        Returns:
            Validity scores (B, 1) - probability of being real
        """
        # Embed labels and reshape to match image dimensions
        label_embed = self.label_embedding(labels)
        label_embed = label_embed.view(-1, 1, 32, 32)
        
        # Concatenate image and label embedding
        x = torch.cat([images, label_embed], dim=1)
        
        # First layer (no normalization)
        first_layer = self.layers[0]
        x = first_layer['conv'](x)
        x = first_layer['activation'](x)
        x = first_layer['dropout'](x)
        x = first_layer['residual'](x)
        
        # Middle layers (with normalization)
        for layer in self.layers[1:]:
            x = layer['conv'](x)
            x = layer['norm'](x)
            x = layer['activation'](x)
            x = layer['dropout'](x)
            x = layer['residual'](x)
        
        # Final classification
        x = self.final(x)
        x = self.sigmoid(x)
        x = x.view(-1, 1)
        
        return x
    
    def arch_parameters(self):
        """
        Return all architecture parameters (α) for optimization
        
        Returns:
            list: All α parameters from mixed operations
        """
        return collect_architecture_parameters(self)
    
    def derive_architecture(self):
        """
        Extract discrete architecture configuration from learned α values
        
        Returns:
            dict: Discrete architecture configuration
        """
        # Use the second layer's choices (first has no norm)
        if len(self.layers) > 1:
            layer = self.layers[1]
            has_norm = True
        else:
            layer = self.layers[0]
            has_norm = False
        
        config = {
            'num_classes': self.num_classes,
            'activation': layer['activation'].derive_discrete_choice(),
            'dropout_rate': layer['dropout'].derive_discrete_choice(),
            'use_residual': layer['residual'].derive_discrete_choice()
        }
        
        # Add norm type if applicable
        if has_norm:
            config['norm_type'] = layer['norm'].derive_discrete_choice()
        else:
            config['norm_type'] = 'batch'  # Default
        
        return config
    
    def get_architecture_weights(self):
        """
        Get current softmax weights for all mixed operations
        
        Returns:
            dict: Current weights for each mixed operation
        """
        weights = {}
        
        for i, layer in enumerate(self.layers):
            if 'norm' in layer:
                weights[f'layer_{i}_norm'] = layer['norm'].get_weights()
            weights[f'layer_{i}_activation'] = layer['activation'].get_weights()
            weights[f'layer_{i}_dropout'] = layer['dropout'].get_weights()
            weights[f'layer_{i}_residual'] = layer['residual'].get_weights()
        
        return weights
