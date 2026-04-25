"""
Architecture Evaluation Module(evaluator.py)

This module provides comprehensive evaluation of GAN architectures,
including training stability, generation quality, and performance metrics.
"""

import torch
import numpy as np
from scipy.ndimage import laplace, label

def evaluate_architecture(generator, discriminator, config, training_stats):
    """
    Basic architecture evaluation
    
    Args:
        generator: Trained generator model
        discriminator: Trained discriminator model
        config: Architecture configuration
        training_stats: Training statistics dictionary
        
    Returns:
        dict: Evaluation metrics
    """
    stability_score = training_stats['stable_epochs'] / 10.0
    
    if training_stats['diverged']:
        return {
            'overall_score': 0.0,
            'stability_score': 0.0,
            'loss_balance_score': 0.0,
            'generation_quality_score': 0.0,
            'training_converged': False
        }
    
    # Loss balance score
    final_g_loss = training_stats['final_g_loss']
    final_d_loss = training_stats['final_d_loss']
    
    g_loss_score = max(0, 1.0 - abs(final_g_loss - 1.0))
    d_loss_score = max(0, 1.0 - abs(final_d_loss - 1.0))
    loss_balance_score = (g_loss_score + d_loss_score) / 2.0
    
    # Generation quality score
    device = next(generator.parameters()).device
    generator.eval()
    with torch.no_grad():
        latent_dim = config['generator']['latent_dim']
        num_samples = 10
        
        quality_scores = []
        for class_id in range(10):
            noise = torch.randn(num_samples, latent_dim).to(device)
            labels = torch.full((num_samples,), class_id, dtype=torch.long).to(device)
            
            fake_images = generator(noise, labels)
            
            # Pixel intensity score
            mean_intensity = torch.mean(torch.abs(fake_images)).item()
            intensity_score = max(0, 1.0 - abs(mean_intensity - 0.5))
            
            # Variation score
            std_intensity = torch.std(fake_images).item()
            variation_score = min(1.0, std_intensity * 2.0)
            
            quality_scores.append((intensity_score + variation_score) / 2.0)
        
        generation_quality_score = np.mean(quality_scores)
    
    # Combined score
    overall_score = (
        0.4 * stability_score +
        0.3 * loss_balance_score +
        0.3 * generation_quality_score
    )
    
    return {
        'overall_score': overall_score,
        'stability_score': stability_score,
        'loss_balance_score': loss_balance_score,
        'generation_quality_score': generation_quality_score,
        'training_converged': True,
        'final_g_loss': final_g_loss,
        'final_d_loss': final_d_loss
    }


class EnsembleEvaluator:
    """
    Enhanced Ensemble Evaluation - Multiple judges for comprehensive assessment
    """
    
    def __init__(self, weights=None):
        self.weights = weights or {
            'base_score': 0.3,
            'diversity_score': 0.2,
            'class_consistency': 0.2,
            'efficiency_score': 0.15,
            'nepali_score': 0.15
        }
    
    def evaluate_architecture_advanced(self, generator, discriminator, config, training_stats):
        """Comprehensive architecture evaluation"""
        device = next(generator.parameters()).device
        results = {}
        
        # Base evaluation
        base_eval = evaluate_architecture(generator, discriminator, config, training_stats)
        results.update(base_eval)
        base_score = base_eval['overall_score']
        
        # Diversity score
        diversity_score = self._evaluate_diversity(generator, config, device)
        results['diversity_score'] = diversity_score
        
        # Class consistency
        class_consistency = self._evaluate_class_consistency(generator, config, device)
        results['class_consistency'] = class_consistency
        
        # Efficiency score
        efficiency_score = self._evaluate_efficiency(training_stats, config)
        results['efficiency_score'] = efficiency_score
        
        # Nepali-specific score
        nepali_score = self._evaluate_nepali_quality(generator, config, device)
        results['nepali_score'] = nepali_score
        
        # Enhanced combined score
        enhanced_score = (
            self.weights['base_score'] * base_score +
            self.weights['diversity_score'] * diversity_score +
            self.weights['class_consistency'] * class_consistency +
            self.weights['efficiency_score'] * efficiency_score +
            self.weights['nepali_score'] * nepali_score
        )
        
        results['enhanced_score'] = enhanced_score
        results['score_breakdown'] = {
            'base': base_score,
            'diversity': diversity_score,
            'class_consistency': class_consistency,
            'efficiency': efficiency_score,
            'nepali_quality': nepali_score
        }
        
        return results
    
    def _evaluate_diversity(self, generator, config, device):
        """Evaluate sample diversity"""
        try:
            generator.eval()
            with torch.no_grad():
                diversity_scores = []
                latent_dim = config['generator']['latent_dim']
                
                for class_label in range(10):
                    noise = torch.randn(16, latent_dim).to(device)
                    labels = torch.full((16,), class_label).to(device)
                    samples = generator(noise, labels)
                    
                    samples_flat = samples.view(samples.size(0), -1)
                    pairwise_distances = torch.pdist(samples_flat).mean().item()
                    diversity_scores.append(pairwise_distances)
                
                avg_diversity = np.mean(diversity_scores)
                return min(1.0, avg_diversity / 2.0)
        except Exception:
            return 0.1
    
    def _evaluate_class_consistency(self, generator, config, device):
        """Evaluate per-class generation consistency"""
        try:
            generator.eval()
            with torch.no_grad():
                class_scores = []
                latent_dim = config['generator']['latent_dim']
                
                for class_label in range(10):
                    batch_scores = []
                    for _ in range(3):
                        noise = torch.randn(8, latent_dim).to(device)
                        labels = torch.full((8,), class_label).to(device)
                        samples = generator(noise, labels)
                        
                        sample_variance = torch.var(samples).item()
                        consistency_score = 1.0 / (1.0 + sample_variance)
                        batch_scores.append(consistency_score)
                    
                    class_scores.append(np.mean(batch_scores))
                
                return np.mean(class_scores)
        except Exception:
            return 0.1
    
    def _evaluate_efficiency(self, training_stats, config):
        """Evaluate training efficiency"""
        try:
            final_g_loss = training_stats.get('final_g_loss', float('inf'))
            final_d_loss = training_stats.get('final_d_loss', float('inf'))
            
            loss_stability = 1.0 / (1.0 + abs(final_g_loss - final_d_loss))
            
            # Get latent_dim from generator config
            latent_dim = config.get('generator', {}).get('latent_dim', 100)
            size_penalty = 1.0 / (1.0 + latent_dim / 100.0)
            
            return (loss_stability + size_penalty) / 2.0
        except Exception:
            return 0.5
    
    def _evaluate_nepali_quality(self, generator, config, device):
        """
        Final, robust evaluation for Nepali digits, validated by testing.
        This version uses a multiplicative approach to ensure that both sharpness
        and structural integrity must be high for a good score.
        """
        try:
            generator.eval()
            with torch.no_grad():
                latent_dim = config['generator']['latent_dim']
                noise = torch.randn(32, latent_dim).to(device)
                labels = torch.randint(0, 10, (32,)).to(device)
                samples = generator(noise, labels)
                
                final_scores = []
                
                for sample in samples:
                    img = sample.squeeze().cpu().numpy()
                    
                    # --- Metric 1: Sharpness Score ---
                    sharpness = np.var(laplace(img))
                    sharpness_score = min(1.0, sharpness / 0.15)
                    
                    # --- Metric 2: Stroke Integrity Score ---
                    binary_stroke = img > 0.1
                    labeled_stroke, num_components = label(binary_stroke)
                    
                    if num_components == 0:
                        structure_score = 0.0
                    elif num_components == 1:
                        structure_score = 1.0
                    elif num_components == 2:
                        structure_score = 0.9
                    else:
                        penalty = max(0.0, 1.0 - (num_components - 2) * 0.25)
                        structure_score = penalty

                    # --- FINAL COMBINATION (THE CRITICAL FIX) ---
                    # We multiply the scores. If either score is zero (e.g., for a
                    # blank image or a completely fragmented one), the final score
                    # is correctly driven to zero. This prevents noise from ever
                    # achieving a high score.
                    final_score = sharpness_score * structure_score
                    final_scores.append(final_score)
                
                # The overall quality is the average of the individual final scores
                return np.mean(final_scores)
                
        except Exception as e:
            print(f"Error in _evaluate_nepali_quality: {e}")
            return 0.1