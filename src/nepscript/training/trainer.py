"""
GAN Training Module

This module provides the GANTrainer class for training Conditional DCGANs
with comprehensive logging, stability monitoring, and checkpoint management.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision.utils import make_grid, save_image
from ..utils.config import resolve_device


class GANTrainer:
    """Trainer class for Conditional DCGAN"""
    
    def __init__(self, generator, discriminator, config, device='auto'):
        """
        Initialize GAN Trainer
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            config: Training configuration dictionary
            device: Device to train on ('auto', 'cuda', 'mps', 'cpu')
        """
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.device = resolve_device(device)
        
        # Training parameters
        self.lr_g = config.get('g_lr', 0.0001)
        self.lr_d = config.get('d_lr', 0.0002)
        self.beta1 = config.get('beta1', 0.5)
        
        
        #freq of discriminator updates
        self.d_update_freq = config.get('d_update_freq',1)   #update d every N batch only
        self.d_batch_counter = 0 #track batches for d update 
        
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizerD = optim.Adam(
            discriminator.parameters(), 
            lr=self.lr_d,
            betas=(self.beta1, 0.999)
        )
        self.optimizerG = optim.Adam(
            generator.parameters(),
            lr=self.lr_g,
            betas=(self.beta1, 0.999)
        )
        
        # Training history
        self.G_losses = []
        self.D_losses = []
        
    def train_epoch(self, train_loader, latent_dim):
        """Train for one epoch"""
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        for i, (real_images, real_labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            real_labels = real_labels.to(self.device)
            
            use_smoothing = self.config.get('label_smoothing', {}).get('enabled', False)
            
            if use_smoothing:
                real_label_value = self.config['label_smoothing'].get('real_label', 0.9)
                fake_label_value = self.config['label_smoothing'].get('fake_label', 0.1)
            else:
                real_label_value = 1.0
                fake_label_value = 0.0
            
            real_label = torch.full((batch_size, 1), real_label_value).to(self.device)
            fake_label = torch.full((batch_size, 1), fake_label_value).to(self.device)
            
            
            
            #generate fake images needed for both d and g
            noise = torch.randn(batch_size, latent_dim).to(self.device)
            fake_labels = torch.randint(0, 10, (batch_size,)).to(self.device)
            fake_images = self.generator(noise, fake_labels)
            
            
            # Train Discriminator
            should_update_d = (self.d_batch_counter%self.d_update_freq ==0)
            
            if should_update_d:
                
                
                self.discriminator.zero_grad()
                
                output_real = self.discriminator(real_images, real_labels)
                lossD_real = self.criterion(output_real, real_label)
                
                
                
                output_fake = self.discriminator(fake_images.detach(), fake_labels)
                lossD_fake = self.criterion(output_fake, fake_label)
                
                lossD = lossD_real + lossD_fake
                lossD.backward()
                self.optimizerD.step()
                
                epoch_d_loss += lossD.item()
            
            else:
                #still compute loss for logging but don't back prop 
                with torch.no_grad():
                    output_real = self.discriminator(real_images, real_labels)
                    lossD_real = self.criterion(output_real, real_label)
                    
                    output_fake = self.discriminator(fake_images.detach(), fake_labels)
                    lossD_fake = self.criterion(output_fake, fake_label)
                    
                    lossD = lossD_real + lossD_fake 
                    epoch_d_loss += lossD.item()
            
            self.d_batch_counter +=1
            
            
            
            # Train Generator
            self.generator.zero_grad()
            output_fake = self.discriminator(fake_images, fake_labels)
            target_for_gen = torch.ones_like(output_fake)
            lossG = self.criterion(output_fake, target_for_gen)
            
            lossG.backward()
            self.optimizerG.step()
            
            epoch_g_loss += lossG.item()
            num_batches += 1
        
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        self.G_losses.append(avg_g_loss)
        self.D_losses.append(avg_d_loss)
        
        return avg_g_loss, avg_d_loss
    
    #  train method for NAS compatibility 
    def train(self, train_loader, num_epochs, latent_dim):
        """
        Train the GAN for multiple epochs (used by NAS Engine).
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            latent_dim: Dimension of latent space
            
        Returns:
            dict: Training statistics
        """
        # Reset losses for this specific NAS run
        self.G_losses = []
        self.D_losses = []
        self.d_batch_counter = 0
        
        
        training_stats = {
            'stable_epochs': 0,
            'diverged': False,
            'final_g_loss': float('inf'),
            'final_d_loss': float('inf'),
            'g_losses': [],
            'd_losses': []
        }
        
        for epoch in range(num_epochs):
            avg_g_loss, avg_d_loss = self.train_epoch(train_loader, latent_dim)
            
            if 0.1 < avg_g_loss < 5.0 and 0.1 < avg_d_loss < 5.0:
                training_stats['stable_epochs'] += 1
            
            if avg_g_loss > 10 or avg_d_loss > 10:
                training_stats['diverged'] = True
                print(f"  - Training diverged at epoch {epoch + 1}")
                break # Stop this run early
        
        training_stats['final_g_loss'] = self.G_losses[-1] if self.G_losses else float('inf')
        training_stats['final_d_loss'] = self.D_losses[-1] if self.D_losses else float('inf')
        training_stats['g_losses'] = self.G_losses
        training_stats['d_losses'] = self.D_losses
        
        return training_stats


    def save_generator_only(self, epoch, save_dir, force_save=False):
        """
        Save only the generator checkpoint (lighter weight).
        
        Args:
            epoch: Current epoch number
            save_dir: Directory to save checkpoint
            force_save: If True, also save as "final" (used for final checkpoint)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save generator state only
        gen_path = save_dir / f"generator_epoch_{epoch:04d}.pth"
        torch.save(self.generator.state_dict(), gen_path)
        
        # If this is a forced save (final checkpoint), also save as "final"
        if force_save:
            final_gen_path = save_dir / "generator_final.pth"
            torch.save(self.generator.state_dict(), final_gen_path)
            print(f"   Final generator saved: {final_gen_path}")
    
    def save_checkpoint(self, epoch, save_dir, force_save=False):
        """
        Save a comprehensive model checkpoint.
        
        Args:
            epoch: Current epoch number
            save_dir: Directory to save checkpoint
            force_save: If True, always save (used for final checkpoint)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizerG.state_dict(),
            'optimizer_D_state_dict': self.optimizerD.state_dict(),
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
            'd_batch_counter': self.d_batch_counter,
        }, checkpoint_path)

        # Always save the generator separately (for easy loading in generation)
        gen_path = save_dir / f"generator_epoch_{epoch:04d}.pth"
        torch.save(self.generator.state_dict(), gen_path)
        
        # If this is a forced save (final checkpoint), also save as "final"
        if force_save:
            final_checkpoint_path = save_dir / "checkpoint_final.pth"
            torch.save({
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_G_state_dict': self.optimizerG.state_dict(),
                'optimizer_D_state_dict': self.optimizerD.state_dict(),
                'G_losses': self.G_losses,
                'D_losses': self.D_losses,
                'd_batch_counter': self.d_batch_counter,
            }, final_checkpoint_path)
            
            final_gen_path = save_dir / "generator_final.pth"
            torch.save(self.generator.state_dict(), final_gen_path)
            
            print(f"   Final checkpoint saved: {final_checkpoint_path}")
            print(f"   Final generator saved: {final_gen_path}")
        
        
    def load_checkpoint(self, checkpoint_path):
        """Load a comprehensive model checkpoint."""
        p = Path(checkpoint_path)
        unified_checkpoint_path = p.parent / p.name.replace("generator_", "checkpoint_")
        
        if not unified_checkpoint_path.exists():
            print(f"Unified checkpoint not found. Loading generator weights only from {p}.")
            self.generator.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            return 0 

        checkpoint = torch.load(unified_checkpoint_path, map_location=self.device, weights_only=True)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.G_losses = checkpoint.get('G_losses', [])
        self.D_losses = checkpoint.get('D_losses', [])
        self.d_batch_counter = checkpoint.get('d_batch_counter',0)
        
        
        print(f"Checkpoint loaded. Resuming with optimizer state and loss history.")
        return checkpoint['epoch']

    def save_sample_images(self, epoch, save_dir, latent_dim, num_samples_per_class=8):
        """Generate and save sample images"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator.eval()
        with torch.no_grad():
            num_classes = 10
            noise = torch.randn(num_samples_per_class * num_classes, latent_dim, device=self.device)
            labels = torch.LongTensor([i for i in range(num_classes) for _ in range(num_samples_per_class)]).to(self.device)
            
            fake_images = self.generator(noise, labels).cpu()
            image_path = save_dir / f"epoch_{epoch:04d}.png"
            grid = make_grid(fake_images, nrow=num_samples_per_class, normalize=True)
            save_image(grid, image_path)
        
        self.generator.train()