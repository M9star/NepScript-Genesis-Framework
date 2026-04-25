"""
SearchableGenerator - Conditional DCGAN Generator with Searchable Hyperparameters

This module contains the SearchableGenerator class that implements a Conditional
Deep Convolutional GAN generator with searchable hyperparameters while maintaining
the proven DCGAN architecture structure.
"""

import torch
import torch.nn as nn
from .blocks import GeneratorResBlock


class DeconvBlock(nn.Module):
    """Configurable deconvolutional block for generator"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm_type='batch', activation='relu', dropout=0.0):
        super(DeconvBlock, self).__init__()
        
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        
        # Normalization
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm_type == 'layer':
            layers.append(nn.GroupNorm(1, out_channels))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'swish':
            layers.append(nn.SiLU(inplace=True))
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class SearchableGenerator(nn.Module):
    """Conditional Generator with FIXED DCGAN architecture, searchable hyperparameters only"""
    
    def __init__(self, config):
        super(SearchableGenerator, self).__init__()
        
        # Configuration parameters - ARCHITECTURE FIXED, only hyperparameters vary
        self.latent_dim = config.get('latent_dim', 100)
        self.num_classes = config.get('num_classes', 10)
        
        # FIXED DCGAN ARCHITECTURE - Standard proven structure (modified for 32x32 output)
        self.channels = config.get('channels', [512, 256, 128])
        
        # SEARCHABLE HYPERPARAMETERS ONLY
        self.norm_type = config.get('norm_type', 'batch')
        self.activation = config.get('activation', 'relu')
        self.dropout_rate = config.get('dropout_rate', 0.0)
        self.use_residual = config.get('use_residual', False)  # NEW: ResBlock option
        
        # Label embedding
        self.label_embedding = nn.Embedding(self.num_classes, self.latent_dim)
        
        # FIXED Initial projection - Standard DCGAN
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim * 2, self.channels[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(True)
        )
        
        # FIXED DCGAN Layer Structure - Only hyperparameters change
        self.layers = nn.ModuleList()
        
        for i in range(len(self.channels) - 1):
            layer = DeconvBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=4,  # FIXED - Standard DCGAN
                stride=2,       # FIXED - Standard DCGAN  
                padding=1,      # FIXED - Standard DCGAN
                norm_type=self.norm_type,    # SEARCHABLE
                activation=self.activation,   # SEARCHABLE
                dropout=self.dropout_rate if i < len(self.channels) - 2 else 0.0  # SEARCHABLE
            )
            self.layers.append(layer)
        
        # ResBlocks - Added after each upsampling layer (except final)
        self.res_blocks = nn.ModuleList()
        if self.use_residual:
            for i in range(len(self.channels) - 1):
                res_block = GeneratorResBlock(
                    channels=self.channels[i + 1],
                    norm_type=self.norm_type,
                    activation=self.activation
                )
                self.res_blocks.append(res_block)
        
        # FIXED Final output layer - Standard DCGAN
        self.final = nn.ConvTranspose2d(self.channels[-1], 1, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, noise, labels):
        # FIXED forward pass - Standard DCGAN flow
        label_embed = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise.unsqueeze(2).unsqueeze(3), label_embed], dim=1)
        
        # Initial projection
        x = self.initial(x)
        
        # Forward through FIXED layer structure with optional ResBlocks
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ResBlock after upsampling (if enabled)
            if self.use_residual:
                x = self.res_blocks[i](x)
        
        # Final output
        x = self.final(x)
        return self.tanh(x)


# Generator search space - ONLY hyperparameters, NO layer changes
GENERATOR_SEARCH_SPACE = {
    'latent_dim': [64, 100, 128, 200],       # Input noise dimension
    'norm_type': ['batch', 'instance'],      # Normalization strategy  
    'activation': ['relu', 'leaky_relu'],    # Activation functions
    'dropout_rate': [0.0, 0.1, 0.2, 0.3],   # Regularization strength
    'use_residual': [False, True]            # Enable residual connections
}