"""
Neural Architecture Search Strategies

This module implements different strategies for searching through the space
of GAN architectures to find optimal configurations for Nepali digit generation.
"""

import random
import copy
import time
import numpy as np
from functools import reduce
import operator
from ..models.factory import sample_architecture_pair
from ..models.generator import GENERATOR_SEARCH_SPACE
from ..models.discriminator import DISCRIMINATOR_SEARCH_SPACE


class AdaptiveRandomSearch:
    """
    Adaptive Random Search - Learns from successful configurations
    
    - First few attempts: Pure exploration (random sampling)
    - After finding good configs: 70% mutation of successful configs, 30% pure exploration
    - Focuses search around promising regions
    """
    
    def __init__(self, adaptation_threshold=0.6, mutation_rate=0.7, min_exploration=3):
        self.successful_configs = []
        self.adaptation_threshold = adaptation_threshold
        self.mutation_rate = mutation_rate
        self.min_exploration = min_exploration
        self.iteration = 0
        
    def sample_config(self):
        """Sample architecture configuration using adaptive strategy"""
        self.iteration += 1
        
        # Early exploration phase
        if len(self.successful_configs) < self.min_exploration:
            config = sample_architecture_pair()
            config['sampling_strategy'] = 'adaptive_pure_exploration'
            return config
        
        # Adaptive phase
        if np.random.random() < self.mutation_rate:
            base_config = random.choice(self.successful_configs)
            config = self._mutate_config(base_config)
            config['sampling_strategy'] = 'adaptive_mutation'
        else:
            config = sample_architecture_pair()
            config['sampling_strategy'] = 'adaptive_continued_exploration'
        
        return config
    
    def _mutate_config(self, base_config):
        """Create a mutation of a successful configuration"""
        config = copy.deepcopy(base_config)
        
        if 'sampling_strategy' in config:
            del config['sampling_strategy']
        
        # Decide randomly to mutate gen and/or disc
        mutate_gen = (random.random() < 0.75)
        mutate_disc = (random.random() < 0.5)
        if not mutate_gen and not mutate_disc:
            mutate_gen = True

        if mutate_gen:
            gen_params = list(GENERATOR_SEARCH_SPACE.keys())
            param_to_mutate = random.choice(gen_params)
            new_value = random.choice(GENERATOR_SEARCH_SPACE[param_to_mutate])
            while new_value == config['generator'][param_to_mutate] and len(GENERATOR_SEARCH_SPACE[param_to_mutate]) > 1:
                new_value = random.choice(GENERATOR_SEARCH_SPACE[param_to_mutate])
            config['generator'][param_to_mutate] = new_value
            
        if mutate_disc:
            disc_params = list(DISCRIMINATOR_SEARCH_SPACE.keys())
            param_to_mutate = random.choice(disc_params)
            new_value = random.choice(DISCRIMINATOR_SEARCH_SPACE[param_to_mutate])
            while new_value == config['discriminator'][param_to_mutate] and len(DISCRIMINATOR_SEARCH_SPACE[param_to_mutate]) > 1:
                new_value = random.choice(DISCRIMINATOR_SEARCH_SPACE[param_to_mutate])
            config['discriminator'][param_to_mutate] = new_value
            
        return config
    
    def update_with_result(self, config, score):
        """Update strategy with architecture results"""
        if score >= self.adaptation_threshold:
            pure_success_config = {
                'id': config['id'],
                'generator': config['generator'],
                'discriminator': config['discriminator']
            }
            
            if pure_success_config not in self.successful_configs:
                self.successful_configs.append(pure_success_config)
            
            # Keep only the best configs
            if len(self.successful_configs) > 10:
                self.successful_configs = self.successful_configs[-10:]
    
    def get_stats(self):
        """Get statistics about the adaptive search"""
        return {
            'successful_configs_count': len(self.successful_configs),
            'iterations': self.iteration,
            'adaptation_active': len(self.successful_configs) >= self.min_exploration
        }


class ProgressiveSearch:
    """
    Dynamic Progressive Search Strategy
    
    Calculates the size of each search phase and dynamically allocates the
    total architecture budget.
    """
    
    def __init__(self, num_architectures):
        self.total_architectures = num_architectures
        self.current_phase = 1
        self.architectures_in_phase = 0
        
        # Define search spaces for each phase
        self.phase_spaces = {
            1: {  # Conservative
                'latent_dim': [100], 'norm_type': ['batch'], 'activation': ['relu'],
                'dropout_rate': [0.0, 0.1], 'use_residual': [False],
                'd_norm_type': ['batch'], 'd_activation': ['leaky_relu'], 
                'd_dropout_rate': [0.0, 0.1], 'd_use_residual': [False]
            },
            2: {  # Moderate
                'latent_dim': [64, 100, 128], 'norm_type': ['batch', 'instance'],
                'activation': ['relu', 'leaky_relu'], 'dropout_rate': [0.0, 0.1, 0.2],
                'use_residual': [False, True],
                'd_norm_type': ['batch', 'instance'], 'd_activation': ['leaky_relu'],
                'd_dropout_rate': [0.0, 0.1, 0.2], 'd_use_residual': [False, True]
            },
            3: {  # Aggressive
                'latent_dim': [64, 100, 128, 200], 'norm_type': ['batch', 'instance'],
                'activation': ['relu', 'leaky_relu'],
                'dropout_rate': [0.0, 0.1, 0.2, 0.3], 'use_residual': [False, True],
                'd_norm_type': ['batch', 'instance'],
                'd_activation': ['leaky_relu', 'relu'],
                'd_dropout_rate': [0.0, 0.2, 0.3, 0.5], 'd_use_residual': [False, True]
            }
        }
        
        self.phase_space_sizes = self._calculate_phase_space_sizes()
        self.phase_targets = self._calculate_dynamic_phase_targets()
        
    def _calculate_phase_space_sizes(self):
        """Calculates the total number of unique architectures in each phase"""
        sizes = {}
        for phase, space in self.phase_spaces.items():
            gen_keys = ['latent_dim', 'norm_type', 'activation', 'dropout_rate', 'use_residual']
            disc_keys = ['d_norm_type', 'd_activation', 'd_dropout_rate', 'd_use_residual']
            
            gen_combos = reduce(operator.mul, [len(space[k]) for k in gen_keys])
            disc_combos = reduce(operator.mul, [len(space[k]) for k in disc_keys])
            
            sizes[phase] = gen_combos * disc_combos
        return sizes

    def _calculate_dynamic_phase_targets(self):
        """Distributes the total architecture budget across phases dynamically"""
        targets = {}
        remaining_architectures = self.total_architectures
        remaining_phases = len(self.phase_spaces)
        
        for phase_num in sorted(self.phase_spaces.keys()):
            if remaining_phases == 0:
                break
            
            ideal_share = remaining_architectures // remaining_phases
            actual_size = self.phase_space_sizes[phase_num]
            
            target_for_this_phase = min(ideal_share, actual_size)
            targets[phase_num] = target_for_this_phase
            
            remaining_architectures -= target_for_this_phase
            remaining_phases -= 1

        if remaining_architectures > 0 and len(targets) > 0:
            last_phase = max(targets.keys())
            targets[last_phase] += remaining_architectures
            
        return targets

    def get_current_phase_target(self):
        """Returns the dynamically calculated target for the current phase"""
        return self.phase_targets.get(self.current_phase, 0)

    def sample_config(self):
        """Sample configuration based on current phase"""
        search_space = self.phase_spaces[self.current_phase]
        
        gen_config = {
            'latent_dim': random.choice(search_space['latent_dim']),
            'num_classes': 10,
            'norm_type': random.choice(search_space['norm_type']),
            'activation': random.choice(search_space['activation']),
            'dropout_rate': random.choice(search_space['dropout_rate']),
            'use_residual': random.choice(search_space['use_residual'])
        }
        
        disc_config = {
            'num_classes': 10,
            'norm_type': random.choice(search_space['d_norm_type']),
            'activation': random.choice(search_space['d_activation']),
            'dropout_rate': random.choice(search_space['d_dropout_rate']),
            'use_residual': random.choice(search_space['d_use_residual'])
        }
        
        self.architectures_in_phase += 1
        temp_arch_id = f"progressive_p{self.current_phase}_{self.architectures_in_phase}"
        
        return {
            'id': temp_arch_id,
            'generator': gen_config,
            'discriminator': disc_config,
            'sampling_strategy': f'progressive_phase_{self.current_phase}'
        }
    
    def advance_phase(self):
        """Move to next phase if available"""
        if self.current_phase < len(self.phase_spaces):
            self.current_phase += 1
            self.architectures_in_phase = 0
            return True
        return False
    
    def get_current_phase_space_size(self):
        """Returns the total number of unique architectures in current phase"""
        return self.phase_space_sizes.get(self.current_phase, 0)
    
    def get_current_phase(self):
        """Get current phase information"""
        phase_names = {1: 'Conservative', 2: 'Moderate', 3: 'Aggressive'}
        return {
            'phase': self.current_phase,
            'phase_name': phase_names[self.current_phase],
            'architectures_in_phase': self.architectures_in_phase
        }


class MultiFidelitySearch:
    """
    Multi-Fidelity Search Strategy - Smart time management
    
    Like a job interview process - quick screening first, then deeper evaluation
    """
    
    def __init__(self, quick_epochs=3, medium_epochs=6, full_epochs=12,
                 quick_threshold=0.5, medium_threshold=0.7):
        self.quick_epochs = quick_epochs
        self.medium_epochs = medium_epochs  
        self.full_epochs = full_epochs
        self.quick_threshold = quick_threshold
        self.medium_threshold = medium_threshold
        self.evaluation_history = []
    
    def sample_config(self):
        """Sample configuration for multi-fidelity search"""
        config = sample_architecture_pair()
        config['sampling_strategy'] = 'multifidelity'
        return config
    
    
    def get_training_epochs(self, previous_score=None, current_stage='quick'):
        """Determine training epochs based on multi-fidelity strategy"""
        if current_stage == 'quick':
            return self.quick_epochs
        elif current_stage == 'medium':
            if previous_score is None or previous_score >= self.quick_threshold:
                return self.medium_epochs
            else:
                return 0
        elif current_stage == 'full':
            if previous_score is None or previous_score >= self.medium_threshold:
                return self.full_epochs
            else:
                return 0
        
        return self.quick_epochs
    
    def should_continue_evaluation(self, score, stage):
        """Decide if architecture should continue to next fidelity level"""
        if stage == 'quick':
            return score >= self.quick_threshold
        elif stage == 'medium':
            return score >= self.medium_threshold
        return False
    
    def record_evaluation(self, config_id, stage, score, epochs):
        """Record evaluation results for analysis"""
        self.evaluation_history.append({
            'config_id': config_id,
            'stage': stage,
            'score': score,
            'epochs': epochs
        })
    
    def get_efficiency_stats(self):
        """Get statistics about computational efficiency"""
        if not self.evaluation_history:
            return {}
        
        total_epochs = sum(record['epochs'] for record in self.evaluation_history)
        stage_counts = {}
        for record in self.evaluation_history:
            stage = record['stage']
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        unique_configs = len(set(record['config_id'] for record in self.evaluation_history))
        
        return {
            'total_epochs_used': total_epochs,
            'evaluations_by_stage': stage_counts,
            'avg_epochs_per_architecture': total_epochs / unique_configs if unique_configs > 0 else 0
        }