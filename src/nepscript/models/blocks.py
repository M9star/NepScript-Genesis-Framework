"""
Residual Blocks for Generator and Discriminator

This module implements residual blocks (ResBlocks) that can be used in both
the generator and discriminator to improve gradient flow and training stability.
"""

import torch.nn as nn


class GeneratorResBlock(nn.Module):
    """
    Residual Block for Generator
    
    Maintains spatial dimensions while learning residual transformations.
    Used after upsampling layers to refine features while preserving gradient flow.
    
    Architecture:
        Input → Conv3x3 → Norm → Activation → Conv3x3 → Norm → (+) → Activation → Output
          |_____________________________________________________________|
                                 Skip Connection
    """
    
    def __init__(self, channels, norm_type='batch', activation='relu'):
        """
        Args:
            channels: Number of input/output channels
            norm_type: Type of normalization ('batch', 'instance', 'none')
            activation: Activation function ('relu', 'leaky_relu')
        """
        super(GeneratorResBlock, self).__init__()
        
        # Main transformation path
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Normalization
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm2d(channels)
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels)
        else:
            self.norm1 = nn.Identity()
        
        # Activation
        if activation == 'relu':
            self.activation1 = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Normalization
        if norm_type == 'batch':
            self.norm2 = nn.BatchNorm2d(channels)
        elif norm_type == 'instance':
            self.norm2 = nn.InstanceNorm2d(channels)
        else:
            self.norm2 = nn.Identity()
        
        # Final activation after skip connection
        if activation == 'relu':
            self.activation2 = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Output tensor [batch, channels, height, width] (same shape as input)
        """
        identity = x  # Save input for skip connection
        
        # Main transformation path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Add skip connection (gradient highway!)
        out = out + identity
        
        # Final activation
        out = self.activation2(out)
        
        return out


class DiscriminatorResBlock(nn.Module):
    """
    Residual Block for Discriminator
    
    Maintains spatial dimensions while learning residual transformations.
    Used after downsampling layers to extract better features while preserving gradient flow.
    
    Architecture:
        Input → Conv3x3 → Norm → Activation → Conv3x3 → Norm → (+) → Activation → Output
          |_____________________________________________________________|
                                 Skip Connection
    """
    
    def __init__(self, channels, norm_type='batch', activation='leaky_relu'):
        """
        Args:
            channels: Number of input/output channels
            norm_type: Type of normalization ('batch', 'instance', 'none')
            activation: Activation function ('relu', 'leaky_relu')
        """
        super(DiscriminatorResBlock, self).__init__()
        
        # Main transformation path
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Normalization
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm2d(channels)
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels)
        else:
            self.norm1 = nn.Identity()
        
        # Activation
        if activation == 'leaky_relu':
            self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation1 = nn.ReLU(inplace=True)
        else:
            self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Normalization
        if norm_type == 'batch':
            self.norm2 = nn.BatchNorm2d(channels)
        elif norm_type == 'instance':
            self.norm2 = nn.InstanceNorm2d(channels)
        else:
            self.norm2 = nn.Identity()
        
        # Final activation after skip connection
        if activation == 'leaky_relu':
            self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation2 = nn.ReLU(inplace=True)
        else:
            self.activation2 = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Output tensor [batch, channels, height, width] (same shape as input)
        """
        identity = x  # Save input for skip connection
        
        # Main transformation path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Add skip connection (gradient highway!)
        out = out + identity
        
        # Final activation
        out = self.activation2(out)
        
        return out
