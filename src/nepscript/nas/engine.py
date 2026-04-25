"""
NAS Engine - Main Orchestrator

This module coordinates the entire NAS process, managing search strategies,
training, evaluation, and result tracking.
"""

import time
import torch
from pathlib import Path
from ..models.factory import create_models_from_config, weights_init, calculate_model_size
from ..training.trainer import GANTrainer
from .evaluator import evaluate_architecture, EnsembleEvaluator
from .strategies import AdaptiveRandomSearch, ProgressiveSearch, MultiFidelitySearch
from .adversarial_nas import AdversarialNASEngine


class NASEngine:
    """Main NAS orchestrator"""
    
    def __init__(self, strategy='adaptive', config=None, device='cuda'):
        """
        Initialize NAS Engine
        
        Args:
            strategy: Search strategy ('adaptive', 'progressive', 'multifidelity', 'random', 'adversarial')
            config: Configuration dictionary
            device: Device to run on
        """
        self.strategy_name = strategy
        self.config = config or {}
        self.device = device
        
        # Initialize search strategy
        self.searcher = self._init_strategy(strategy)
        self.evaluator = EnsembleEvaluator() #if strategy != 'random' else None
        
        # Results tracking
        self.results = []
        self.best_score = 0.0
        self.best_config = None
        self.tested_architectures = set()
        
    def _init_strategy(self, strategy):
        """Initialize the appropriate search strategy"""
        if strategy == 'adaptive':
            return AdaptiveRandomSearch()
        elif strategy == 'progressive':
            num_archs = self.config.get('num_architectures', 10)
            return ProgressiveSearch(num_architectures=num_archs)
        elif strategy == 'multifidelity':
            multifidelity_config = self.config.get('multifidelity', {})
            return MultiFidelitySearch(
            quick_epochs=multifidelity_config.get('quick_epochs', 10),
            medium_epochs=multifidelity_config.get('medium_epochs', 20),
            full_epochs=multifidelity_config.get('full_epochs', 30),
            quick_threshold=multifidelity_config.get('quick_threshold', 0.5),
            medium_threshold=multifidelity_config.get('medium_threshold', 0.7))
            # return MultiFidelitySearch()
        elif strategy == 'adversarial':
            # AdversarialNAS uses its own run_search implementation
            return AdversarialNASEngine(
                config=self.config,
                device=self.device
            )
        else:
            return None
    
    def run_search(self, train_loader, num_architectures, base_epochs):
        """
        Run the NAS experiment
        
        Args:
            train_loader: DataLoader for training data
            num_architectures: Number of architectures to evaluate
            base_epochs: Base number of training epochs
            
        Returns:
            tuple: (results, best_config)
        """
        print(f" Starting {self.strategy_name.upper()} NAS Experiment")
        print(f"Target architectures: {num_architectures}")
        print(f"Base epochs: {base_epochs}")
        print("=" * 60)
        
        # Handle adversarial strategy separately
        if self.strategy_name == 'adversarial':
            return self._run_adversarial_search(train_loader, base_epochs)
        
        tested_count = 0
        unique_in_phase_count = 0
        
        while tested_count < num_architectures:
            print(f"\n[{tested_count + 1}/{num_architectures}] Sampling new architecture...")
            
            try:
                # Sample configuration
                if self.searcher:
                    config = self.searcher.sample_config()
                else:
                    from ..models.factory import sample_architecture_pair
                    config = sample_architecture_pair()
                
                # Generate content-based ID
                gen_hash = hash(str(config['generator']))
                disc_hash = hash(str(config['discriminator']))
                content_id = f"G{gen_hash % 10000:04d}_D{disc_hash % 10000:04d}"
                
                # Check for duplicates
                if content_id in self.tested_architectures:
                    print(f" Duplicate architecture ({content_id}). Skipping...")
                    
                    if self.strategy_name == 'progressive':
                        self._handle_progressive_phase(unique_in_phase_count)
                    continue
                
                # Record and standardize ID
                self.tested_architectures.add(content_id)
                config['id'] = content_id
                
                # Handle progressive phase advancement
                if self.strategy_name == 'progressive':
                    unique_in_phase_count += 1
                    phase_advanced = self._handle_progressive_phase(unique_in_phase_count)
                    if phase_advanced:
                        unique_in_phase_count = 0
                
                print(f"Architecture ID: {config['id']}")
                
                # Train and evaluate
                result = self._train_and_evaluate(
                    config, train_loader, base_epochs
                )
                
                self.results.append(result)
                
                # Update best
                current_score = result.get('enhanced_score', result.get('overall_score', 0))
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_config = config
                    print(f"NEW BEST! Score: {self.best_score:.4f}")
                
                # Update adaptive strategy
                if self.strategy_name == 'adaptive':
                    self.searcher.update_with_result(config, current_score)
                
                tested_count += 1
                
            except Exception as e:
                print(f"Architecture failed: {str(e)}")
                tested_count += 1
        
        print(f"\n {self.strategy_name.upper()} NAS completed!")
        print(f"Best: {self.best_config['id'] if self.best_config else 'None'} (Score: {self.best_score:.4f})")
        
        return self.results, self.best_config
    
    def _train_and_evaluate(self, config, train_loader, base_epochs):
        """Train and evaluate a single architecture"""
        # Create models
        generator, discriminator = create_models_from_config(config, self.device)
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        
        # Get model sizes
        gen_size = calculate_model_size(generator)
        disc_size = calculate_model_size(discriminator)
        
        print(f"Model sizes - G: {gen_size:.2f}M, D: {disc_size:.2f}M")
        
        # Train
        if self.strategy_name == 'multifidelity':
            result = self._multifidelity_train(
                generator, discriminator, config, train_loader
            )
        else:
            trainer = GANTrainer(generator, discriminator, self.config, self.device)
            latent_dim = config['generator']['latent_dim']
            
            start_time = time.time()
            training_stats = trainer.train(train_loader, base_epochs, latent_dim)
            training_time = time.time() - start_time
            
            # Evaluate
            if self.evaluator:
                eval_results = self.evaluator.evaluate_architecture_advanced(
                    generator, discriminator, config, training_stats
                )
            else:
                eval_results = evaluate_architecture(
                    generator, discriminator, config, training_stats
                )
            
            result = {
                'architecture_id': config['id'],
                'config': config,
                'strategy_used': self.strategy_name,
                'training_time': training_time,
                'model_size_mb': gen_size + disc_size,
                **eval_results
            }
        
        # Cleanup
        del generator, discriminator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def _multifidelity_train(self, generator, discriminator, config, train_loader):
        """Multi-fidelity training pipeline"""
        current_score = 0.0
        stages = ['quick', 'medium', 'full']
        total_time = 0
        
        for stage in stages:
            epochs = self.searcher.get_training_epochs(current_score, stage)
            if epochs == 0:
                print(f"Skipping {stage} evaluation")
                break
            
            print(f"{stage.title()} evaluation: {epochs} epochs")
            trainer = GANTrainer(generator, discriminator, self.config, self.device)
            latent_dim = config['generator']['latent_dim']
            
            start_time = time.time()
            training_stats = trainer.train(train_loader, epochs, latent_dim)
            total_time += time.time() - start_time
            
            # Evaluate
            if self.evaluator:
                eval_results = self.evaluator.evaluate_architecture_advanced(
                    generator, discriminator, config, training_stats
                )
                current_score = eval_results.get('enhanced_score', eval_results['overall_score'])
            else:
                eval_results = evaluate_architecture(
                    generator, discriminator, config, training_stats
                )
                current_score = eval_results['overall_score']
            
            self.searcher.record_evaluation(config['id'], stage, current_score, epochs)
            print(f"{stage.title()} score: {current_score:.4f}")
            
            if not self.searcher.should_continue_evaluation(current_score, stage):
                break
        
        gen_size = calculate_model_size(generator)
        disc_size = calculate_model_size(discriminator)
        
        return {
            'architecture_id': config['id'],
            'config': config,
            'strategy_used': self.strategy_name,
            'training_time': total_time,
            'model_size_mb': gen_size + disc_size,
            **eval_results
        }
    
    def _handle_progressive_phase(self, unique_count):
        """Handle progressive search phase advancement"""
        phase_info = self.searcher.get_current_phase()
        phase_target = self.searcher.get_current_phase_target()
        
        print(f"Phase: {phase_info['phase_name']}, Unique: {unique_count}/{phase_target}")
        
        if unique_count >= phase_target:
            if self.searcher.advance_phase():
                new_phase = self.searcher.get_current_phase()
                print(f"Advanced to Phase {new_phase['phase']} ({new_phase['phase_name']})")
                return True
        return False
    
    def _run_adversarial_search(self, train_loader, base_epochs):
        """Run adversarial NAS search with gradient-based optimization"""
        print("\n  Gradient-based Architecture Search")
        print("This uses continuous relaxation and differentiable operations.")
        
        # AdversarialNAS needs both train and validation data
        val_loader = None
        try:
            # Use existing load_data function for validation
            from ..utils.data import load_data
            
            data_config = self.config.get('data', {})
            data_dir = data_config.get('data_dir', 'data/DevanagariHandwrittenDigitDataset')
            labels_csv = data_config.get('labels_csv', 'data/hindi_mnist.csv')
            batch_size = data_config.get('batch_size', 32)
            max_subset = data_config.get('max_subset_per_class')
            
            # Try to load Test split as validation
            val_loader, val_size = load_data(
                data_dir=data_dir,
                labels_csv=labels_csv,
                batch_size=batch_size,
                split='Test',  # Use Test split for validation
                max_subset_per_class=max_subset,
                num_workers=data_config.get('num_workers', 0),
                augment=False  # No augmentation for validation
            )
            
            if val_loader is not None:
                print(f"  Using Test split as validation data")
                print(f"   Train samples: {len(train_loader.dataset)}")
                print(f"   Validation samples: {val_size}")
            else:
                raise Exception("Could not load Test split")
                
        except Exception as e:
            # Fallback: Split training data 80/20
            print(f"    Could not load Test split, creating validation split from training data")
            print(f"   Reason: {str(e)}")
            
            from torch.utils.data import random_split
            
            dataset = train_loader.dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = random_split(
                dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=train_loader.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            print(f"  Split training data: {train_size} train, {val_size} validation samples")
        
        # Run adversarial search
        print(f"   Starting search for {base_epochs} epochs...")
        best_config = self.searcher.search(
            train_loader=train_loader,
            val_loader=val_loader,
            search_epochs=base_epochs  # ← Fixed parameter name
        )
        
        # Save search results
        if self.config.get('save_results', True):
            output_dir = Path(self.config.get('output_dir', 'experiments/nas_results'))
            self.searcher.save_search_results(output_dir, best_config)
        
        # Format result for consistency
        result = {
            'architecture_id': best_config.get('id', 'adversarial_best'),
            'config': best_config,
            'strategy_used': 'adversarial',
            'search_type': 'gradient_based',
            'note': 'Architecture found via continuous relaxation'
        }
        
        self.results = [result]
        self.best_config = best_config
        self.best_score = 1.0
        
        print(f"\n  ADVERSARIAL NAS completed!")
        print(f"  Best architecture: {best_config['id']}")
        
        return self.results, self.best_config
