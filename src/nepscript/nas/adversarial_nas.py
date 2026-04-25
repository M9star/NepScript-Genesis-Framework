"""
AdversarialNAS Engine - Gradient-Based Neural Architecture Search for GANs

This module implements the main search algorithm from the AdversarialNAS paper.
It performs three-level optimization:
    Level 1: Discriminator architecture (α_D) on validation data
    Level 2: Generator architecture (α_G) on validation data  
    Level 3: Network weights (w_G, w_D) on training data (standard GAN training)

Reference:
    Gong et al. "AdversarialNAS: Adversarial Neural Architecture Search for GANs"
    CVPR 2019
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from .supernet import DifferentiableGenerator, DifferentiableDiscriminator


class AdversarialNASEngine:
    """
    Main search engine for gradient-based GAN architecture search
    
    The engine coordinates three-level optimization:
    1. Update discriminator architecture (α_D) to improve validation performance
    2. Update generator architecture (α_G) to fool discriminator on validation data
    3. Update network weights (w_G, w_D) via standard GAN training
    
    Args:
        latent_dim: Dimension of input noise vector
        device: Device to run on ('cuda' or 'cpu')
        config: Configuration dictionary with hyperparameters
    """
    
    def __init__(self, latent_dim=100, device='cuda', config=None):
        self.latent_dim = latent_dim
        self.device = device
        self.config = config or {}
        
        # Extract hyperparameters
        self.lr_weights = self.config.get('lr_weights', 0.0002)
        self.lr_arch = self.config.get('lr_arch', 0.001)
        self.warmup_epochs = self.config.get('warmup_epochs', 3)
        self.beta1 = self.config.get('beta1', 0.5)
        self.beta2 = self.config.get('beta2', 0.999)
        
        # Create supernets (contain ALL architectural choices)
        print("Initializing supernets...")
        self.supernet_G = DifferentiableGenerator(latent_dim=latent_dim).to(device)
        self.supernet_D = DifferentiableDiscriminator().to(device)
        
        # Initialize weights
        self.supernet_G.apply(self._weights_init)
        self.supernet_D.apply(self._weights_init)
        
        # Optimizer for network weights (w_G, w_D)
        # Standard GAN training optimizer
        self.optimizer_weights = optim.Adam(
            list(self.supernet_G.parameters()) + 
            list(self.supernet_D.parameters()),
            lr=self.lr_weights,
            betas=(self.beta1, self.beta2)
        )
        
        # Optimizer for architecture parameters (α_G, α_D)
        # Separate optimizer with different learning rate
        arch_params = (
            self.supernet_G.arch_parameters() + 
            self.supernet_D.arch_parameters()
        )
        self.optimizer_arch = optim.Adam(
            arch_params,
            lr=self.lr_arch,
            betas=(self.beta1, self.beta2)
        )
        
        # Loss function (Binary Cross Entropy)
        self.criterion = nn.BCELoss()
        
        # Tracking
        self.search_history = {
            'arch_losses': [],
            'g_losses': [],
            'd_losses': [],
            'arch_weights': []
        }
    
    def _weights_init(self, m):
        """Initialize network weights (DCGAN initialization)"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def search(self, train_loader, val_loader, search_epochs=25):
        """
        Main search loop - performs gradient-based architecture search
        
        Args:
            train_loader: DataLoader for training data (for weight updates)
            val_loader: DataLoader for validation data (for architecture updates)
            search_epochs: Number of epochs to search
            
        Returns:
            dict: Best architecture configuration
        """
        print(f"\n{'='*60}")
        print(f"STARTING ADVERSARIALNAS SEARCH")
        print(f"{'='*60}")
        print(f"Search epochs: {search_epochs}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Learning rate (weights): {self.lr_weights}")
        print(f"Learning rate (architecture): {self.lr_arch}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(search_epochs):
            epoch_start = time.time()
            
            # Phase 1 & 2: Update architecture parameters (after warmup)
            if epoch >= self.warmup_epochs:
                arch_loss = self._update_architecture_params(val_loader)
                self.search_history['arch_losses'].append(arch_loss)
            else:
                print(f"[Epoch {epoch+1}/{search_epochs}] WARMUP - Skipping architecture update")
                self.search_history['arch_losses'].append(0.0)
            
            # Phase 3: Standard GAN training on training data
            g_loss_avg, d_loss_avg = self._train_gan_step(train_loader)
            self.search_history['g_losses'].append(g_loss_avg)
            self.search_history['d_losses'].append(d_loss_avg)
            
            # Record current architecture weights
            arch_weights = {
                'generator': self.supernet_G.get_architecture_weights(),
                'discriminator': self.supernet_D.get_architecture_weights()
            }
            self.search_history['arch_weights'].append(arch_weights)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            if epoch >= self.warmup_epochs:
                print(f"[Epoch {epoch+1}/{search_epochs}] "
                      f"Arch Loss: {arch_loss:.4f} | "
                      f"G Loss: {g_loss_avg:.4f} | "
                      f"D Loss: {d_loss_avg:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            else:
                print(f"[Epoch {epoch+1}/{search_epochs}] (Warmup) "
                      f"G Loss: {g_loss_avg:.4f} | "
                      f"D Loss: {d_loss_avg:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Log architecture evolution every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._log_architecture_evolution()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"SEARCH COMPLETED in {total_time/60:.2f} minutes")
        print(f"{'='*60}\n")
        
        # Derive final architecture
        best_config = self._derive_final_architecture()
        
        return best_config
    
    def _update_architecture_params(self, val_loader):
        """
        Update architecture parameters (α_G, α_D) on validation data
        
        This is Level 1 & 2 of the three-level optimization.
        Goal: Find architectures that generalize well to validation data
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            float: Combined architecture loss
        """
        self.supernet_G.train()
        self.supernet_D.train()
        
        # Use one validation batch per epoch (as per paper)
        val_iter = iter(val_loader)
        try:
            real_imgs, labels = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            real_imgs, labels = next(val_iter)
        
        real_imgs = real_imgs.to(self.device)
        labels = labels.to(self.device)
        batch_size = real_imgs.size(0)
        
        # Generate fake images using current architecture
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_imgs = self.supernet_G(noise, labels)
        
        # --- LEVEL 1: Update discriminator architecture (α_D) ---
        # Compute discriminator validation loss
        real_validity = self.supernet_D(real_imgs, labels)
        fake_validity = self.supernet_D(fake_imgs.detach(), labels)
        
        # Real labels = 1, Fake labels = 0
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        d_loss_real = self.criterion(real_validity, real_labels)
        d_loss_fake = self.criterion(fake_validity, fake_labels)
        d_loss_val = (d_loss_real + d_loss_fake) / 2
        
        # --- LEVEL 2: Update generator architecture (α_G) ---
        # Compute generator validation loss
        fake_validity_for_g = self.supernet_D(fake_imgs, labels)
        g_loss_val = self.criterion(fake_validity_for_g, real_labels)  # Want D to think fakes are real
        
        # Combined architecture loss
        arch_loss = d_loss_val + g_loss_val
        
        # Update architecture parameters only
        self.optimizer_arch.zero_grad()
        arch_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.supernet_G.arch_parameters() + self.supernet_D.arch_parameters(),
            max_norm=1.0
        )
        
        self.optimizer_arch.step()
        
        return arch_loss.item()
    
    def _train_gan_step(self, train_loader):
        """
        Standard GAN training - update network weights (w_G, w_D)
        
        This is Level 3 of the three-level optimization.
        Standard adversarial training on the training set.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            tuple: (avg_g_loss, avg_d_loss)
        """
        self.supernet_G.train()
        self.supernet_D.train()
        
        g_losses = []
        d_losses = []
        
        for batch_idx, (real_imgs, labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(self.device)
            labels = labels.to(self.device)
            batch_size = real_imgs.size(0)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            self.optimizer_weights.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_imgs = self.supernet_G(noise, labels)
            
            # Discriminator loss
            real_validity = self.supernet_D(real_imgs, labels)
            fake_validity = self.supernet_D(fake_imgs.detach(), labels)
            
            d_loss_real = self.criterion(real_validity, real_labels)
            d_loss_fake = self.criterion(fake_validity, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            # Update discriminator
            d_loss.backward()
            self.optimizer_weights.step()
            
            d_losses.append(d_loss.item())
            
            # -----------------
            # Train Generator
            # -----------------
            self.optimizer_weights.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_imgs = self.supernet_G(noise, labels)
            
            # Generator loss (want discriminator to think fakes are real)
            fake_validity = self.supernet_D(fake_imgs, labels)
            g_loss = self.criterion(fake_validity, real_labels)
            
            # Update generator
            g_loss.backward()
            self.optimizer_weights.step()
            
            g_losses.append(g_loss.item())
            
            # Limit batches for faster epochs during search
            if batch_idx >= 50:  # Process ~50 batches per epoch during search
                break
        
        return sum(g_losses) / len(g_losses), sum(d_losses) / len(d_losses)
    
    def _derive_final_architecture(self):
        """
        Extract discrete architecture from learned α parameters
        
        After search completes, this converts the continuous architecture
        parameters to discrete choices.
        
        Returns:
            dict: Complete architecture configuration
        """
        print("\nDeriving final architecture from learned parameters...")
        
        gen_config = self.supernet_G.derive_architecture()
        disc_config = self.supernet_D.derive_architecture()
        
        # Create complete config in our standard format
        config = {
            'id': 'adversarial_nas_derived',
            'generator': gen_config,
            'discriminator': disc_config,
            'sampling_strategy': 'adversarial_nas'
        }
        
        print(f"\nDerived Generator Architecture:")
        for key, value in gen_config.items():
            print(f"  {key}: {value}")
        
        print(f"\nDerived Discriminator Architecture:")
        for key, value in disc_config.items():
            print(f"  {key}: {value}")
        
        return config
    
    def _log_architecture_evolution(self):
        """Log current architecture weights for monitoring"""
        print("\n--- Architecture Evolution ---")
        
        # Generator architecture
        gen_weights = self.supernet_G.get_architecture_weights()
        print("Generator (Layer 0):")
        for key, weights in gen_weights.items():
            if 'layer_0' in key:
                print(f"  {key}: {weights}")
        
        # Discriminator architecture
        disc_weights = self.supernet_D.get_architecture_weights()
        print("Discriminator (Layer 1):")
        for key, weights in disc_weights.items():
            if 'layer_1' in key:
                print(f"  {key}: {weights}")
        
        print("-----------------------------\n")
    
    def save_search_results(self, output_dir, best_config):
        """
        Save search results and history
        
        Args:
            output_dir: Directory to save results
            best_config: Best architecture configuration
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best architecture
        config_path = output_path / "best_architecture.json"
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"Saved best architecture to: {config_path}")
        
        # Save search history
        history_path = output_path / "search_history.json"
        # Convert arch_weights to serializable format
        history_to_save = {
            'arch_losses': self.search_history['arch_losses'],
            'g_losses': self.search_history['g_losses'],
            'd_losses': self.search_history['d_losses']
        }
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=2)
        
        print(f"Saved search history to: {history_path}")
        
        # Save final supernet weights (optional, for analysis)
        supernet_path = output_path / "final_supernet.pth"
        torch.save({
            'generator_state_dict': self.supernet_G.state_dict(),
            'discriminator_state_dict': self.supernet_D.state_dict(),
            'config': best_config
        }, supernet_path)
        
        print(f"Saved final supernet to: {supernet_path}")
