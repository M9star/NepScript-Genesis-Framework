"""
SearchableDiscriminator - Conditional DCGAN Discriminator with Searchable Hyperparameters

This module contains the SearchableDiscriminator class that implements a Conditional
Deep Convolutional GAN discriminator with searchable hyperparameters while maintaining
the proven DCGAN architecture structure.
"""

import torch
import torch.nn as nn
from .blocks import DiscriminatorResBlock


class ConvBlock(nn.Module):
    """Configurable convolutional block for discriminator"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, 
                 norm_type='batch', activation='leaky_relu', dropout=0.0):
        super(ConvBlock, self).__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        
        # Normalization
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm_type == 'layer':
            layers.append(nn.GroupNorm(1, out_channels))
        
        # Activation
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'swish':
            layers.append(nn.SiLU(inplace=True))
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class SearchableDiscriminator(nn.Module):
    """Conditional Discriminator with FIXED DCGAN architecture, searchable hyperparameters only"""
    
    def __init__(self, config):
        super(SearchableDiscriminator, self).__init__()
        
        # Configuration parameters - ARCHITECTURE FIXED, only hyperparameters vary
        self.num_classes = config.get('num_classes', 10)
        
        # FIXED DCGAN ARCHITECTURE - Standard proven structure  
        self.channels = [64, 128, 256]  # FIXED - proven DCGAN channels
        
        # SEARCHABLE HYPERPARAMETERS 
        self.norm_type = config.get('norm_type', 'batch')
        self.activation = config.get('activation', 'leaky_relu')
        self.dropout_rate = config.get('dropout_rate', 0.0)
        self.use_spectral_norm = config.get('use_spectral_norm', False)
        self.use_residual = config.get('use_residual', False)  # NEW: ResBlock option
        
        # Label embedding for conditional discrimination
        self.label_embedding = nn.Embedding(self.num_classes, 32 * 32)
        
        # FIXED DCGAN Layer Structure - Only hyperparameters change
        self.layers = nn.ModuleList()
        
        # FIXED First layer (no normalization) - Standard DCGAN
        first_layer = ConvBlock(
            in_channels=2,  # Image + label embedding - FIXED
            out_channels=self.channels[0],
            kernel_size=4,  # FIXED - Standard DCGAN
            stride=2,       # FIXED - Standard DCGAN
            padding=1,      # FIXED - Standard DCGAN
            norm_type=None, # FIXED - No norm on first layer (DCGAN standard)
            activation=self.activation,    # SEARCHABLE
            dropout=self.dropout_rate      # SEARCHABLE
        )
        self.layers.append(first_layer)
        
        # FIXED Subsequent layers - Standard DCGAN structure
        for i in range(len(self.channels) - 1):
            layer = ConvBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=4,  # FIXED - Standard DCGAN
                stride=2,       # FIXED - Standard DCGAN  
                padding=1,      # FIXED - Standard DCGAN
                norm_type=self.norm_type,    # SEARCHABLE
                activation=self.activation,   # SEARCHABLE
                dropout=self.dropout_rate     # SEARCHABLE
            )
            self.layers.append(layer)
        
        # ResBlocks - Added after each downsampling layer
        self.res_blocks = nn.ModuleList()
        if self.use_residual:
            for i in range(len(self.channels)):
                res_block = DiscriminatorResBlock(
                    channels=self.channels[i],
                    norm_type=self.norm_type if i > 0 else 'batch',  # First layer uses batch norm
                    activation=self.activation
                )
                self.res_blocks.append(res_block)
        
        # FIXED Final classification layer - Standard DCGAN
        self.final = nn.Conv2d(self.channels[-1], 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image, labels):
        # FIXED forward pass - Standard DCGAN flow
        label_embed = self.label_embedding(labels).view(-1, 1, 32, 32)
        
        # Concatenate image and label embedding
        x = torch.cat([image, label_embed], dim=1)
        
        # Forward through FIXED layer structure with optional ResBlocks
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ResBlock after downsampling (if enabled)
            if self.use_residual:
                x = self.res_blocks[i](x)
        
        # Final classification
        x = self.final(x)
        return self.sigmoid(x.view(-1, 1))


# Discriminator search space - ONLY hyperparameters, NO layer changes  
DISCRIMINATOR_SEARCH_SPACE = {
    'norm_type': ['batch', 'instance'],      # Normalization strategy
    'activation': ['leaky_relu', 'relu'],    # Activation functions  
    'dropout_rate': [0.0, 0.2, 0.3, 0.5],   # Regularization strength
    'use_residual': [False, True]            # Enable residual connections
}