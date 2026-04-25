"""
NepScript Genesis - Gradio Web Interface
Complete UI for NAS-GAN Devanagari Digit Generation

Features:
1. Generate digits from trained models
2. Run NAS experiments with different strategies
3. Train models from scratch or resume from checkpoints
4. View and compare results
5. Compare images generted by 2 to 4 models at once 
6. Compare Architecture : tab to visualize the top 3 architecture from all the search strategies  
"""

import gradio as gr
import subprocess
import json
import os
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from datetime import datetime
import glob
import re
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root)) 

def generate_architecture_diagram_html(arch_config):
    """
    Generate HTML diagram for GAN architecture with improved styling.
    
    Args:
        arch_config: Architecture configuration dict with 'generator' and 'discriminator'
    
    Returns:
        str: HTML string with the architecture diagram
    """
    try:
        gen_config = arch_config.get('generator', {})
        disc_config = arch_config.get('discriminator', {})
        
        # Extract hyperparameters with defaults
        latent_dim = gen_config.get('latent_dim', 100)
        gen_norm = gen_config.get('norm_type', 'batch')
        gen_activation = gen_config.get('activation', 'relu')
        gen_dropout = gen_config.get('dropout_rate', 0.0)
        
        disc_norm = disc_config.get('norm_type', 'batch')
        disc_activation = disc_config.get('activation', 'leaky_relu')
        disc_dropout = disc_config.get('dropout_rate', 0.0)
        
        # Capitalize for display
        gen_norm_display = gen_norm.capitalize() + "Norm"
        gen_activation_display = gen_activation.upper().replace('_', ' ')
        disc_norm_display = disc_norm.capitalize() + "Norm"
        disc_activation_display = disc_activation.upper().replace('_', ' ')
        
        
        
        html = f"""
        <style>
            /* --- Start of Scoped CSS --- */
            .arch-container {{
                display: flex;
                gap: 15px; /* MODIFIED: Reduced gap */
                flex-wrap: nowrap; /* MODIFIED: Prevents columns from stacking */
                justify-content: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            
            .arch-container .arch-column {{
                background: white;
                padding: 12px 15px; /* MODIFIED: Reduced padding */
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border: 1px solid #e2e8f0;
                width: 100%;
                max-width: 360px; /* MODIFIED: Reduced max-width to make columns narrower */
                color: #2d3748; 
            }}
            
            .arch-container .arch-title {{
                font-size: 1.0em; /* MODIFIED: Reduced font size */
                font-weight: 600;
                margin-bottom: 12px; /* MODIFIED: Reduced margin */
                color: #1a202c;
                border-bottom: 2px solid #cbd5e0;
                padding-bottom: 8px; /* MODIFIED: Reduced padding */
                text-align: center;
            }}
            
            .arch-container .layer-box {{
                border: 1px solid #cbd5e0;
                padding: 10px; /* MODIFIED: Reduced padding */
                margin: 8px 0; /* MODIFIED: Reduced margin */
                background: #fdfdff;
                font-size: 0.8em; /* MODIFIED: Reduced font size */
                line-height: 1.5; /* MODIFIED: Reduced line height */
                border-radius: 6px;
                color: #2d3748;
            }}
            
            .arch-container .layer-title {{
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 4px; /* MODIFIED: Reduced margin */
            }}
            
            .arch-container .layer-detail {{
                color: #4a5568;
                margin: 2px 0;
                font-family: 'Fira Code', 'Courier New', monospace;
            }}
            
            .arch-container .arrow {{
                text-align: center;
                font-size: 1.1em; /* MODIFIED: Reduced font size */
                color: #a0aec0;
                margin: 1px 0; /* MODIFIED: Reduced margin */
            }}
            
            .arch-container .input-output {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                text-align: center;
                font-weight: 500;
            }}
            
            .arch-container .final-output {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                border: none;
                text-align: center;
                font-weight: 500;
            }}
            
            /* Colored hyperparameter text */
            .arch-container .param-norm {{ color: #38a169; font-weight: 600; }}
            .arch-container .param-activation {{ color: #3182ce; font-weight: 600; }}
            .arch-container .param-dropout {{ color: #dd6b20; font-weight: 600; }}
            .arch-container .param-latent {{ color: #d69e2e; font-weight: 600; }}
            .arch-container .param-channels {{ color: #805ad5; font-weight: 600; }}
            /* --- End of Scoped CSS --- */
        </style>
        
        
        <div class="arch-container">
            <!-- GENERATOR -->
            <div class="arch-column">
                <div class="arch-title">      Generator Architecture</div>
                
                <div class="layer-box input-output">
                    <div>Input</div>
                    <div class="layer-detail" style="color: white; opacity: 0.9;">
                        Noise [<span class="param-latent" style="color: #f6e05e;">{latent_dim}</span>] + Label [10] → Embedding [<span class="param-latent" style="color: #f6e05e;">{latent_dim}</span>]
                    </div>
                    <div class="layer-detail" style="color: white; opacity: 0.9;">
                        Combined: [<span class="param-latent" style="color: #f6e05e;">{latent_dim * 2}</span>]
                    </div>
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box">
                    <div class="layer-title">Initial Projection</div>
                    <div class="layer-detail">ConvTranspose2d({latent_dim * 2} → <span class="param-channels">512</span>)</div>
                    <div class="layer-detail">Output: 4×4×<span class="param-channels">512</span></div>
                    <div class="layer-detail">BatchNorm2d + ReLU</div>
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box">
                    <div class="layer-title">Layer 1: <span class="param-channels">512 → 256</span> channels</div>
                    <div class="layer-detail">ConvTranspose2d(512, 256, k=4, s=2, p=1)</div>
                    <div class="layer-detail">Output: 8×8×<span class="param-channels">256</span></div>
                    <div class="layer-detail">
                        <span class="param-norm">{gen_norm_display}</span> + 
                        <span class="param-activation">{gen_activation_display}</span>
                    </div>
                    {"<div class='layer-detail'>Dropout: <span class='param-dropout'>" + str(gen_dropout) + "</span></div>" if gen_dropout > 0 else ""}
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box">
                    <div class="layer-title">Layer 2: <span class="param-channels">256 → 128</span> channels</div>
                    <div class="layer-detail">ConvTranspose2d(256, 128, k=4, s=2, p=1)</div>
                    <div class="layer-detail">Output: 16×16×<span class="param-channels">128</span></div>
                    <div class="layer-detail">
                        <span class="param-norm">{gen_norm_display}</span> + 
                        <span class="param-activation">{gen_activation_display}</span>
                    </div>
                    {"<div class='layer-detail'>Dropout: <span class='param-dropout'>" + str(gen_dropout) + "</span></div>" if gen_dropout > 0 else ""}
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box">
                    <div class="layer-title">Final Output Layer</div>
                    <div class="layer-detail">ConvTranspose2d(128, 1, k=4, s=2, p=1)</div>
                    <div class="layer-detail">Output: 32×32×1 (Grayscale)</div>
                    <div class="layer-detail">Tanh Activation</div>
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box final-output">
                    Generated Devanagari Digit [32×32]
                </div>
            </div>
            
            <!-- DISCRIMINATOR -->
            <div class="arch-column">
                <div class="arch-title"> Discriminator Architecture</div>
                
                <div class="layer-box input-output">
                    <div>Input</div>
                    <div class="layer-detail" style="color: white; opacity: 0.9;">
                        Image [32×32×1] + Label Embedding [32×32]
                    </div>
                    <div class="layer-detail" style="color: white; opacity: 0.9;">
                        Combined: [32×32×2]
                    </div>
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box">
                    <div class="layer-title">Layer 1: <span class="param-channels">2 → 64</span> channels</div>
                    <div class="layer-detail">Conv2d(2, 64, k=4, s=2, p=1)</div>
                    <div class="layer-detail">Output: 16×16×<span class="param-channels">64</span></div>
                    <div class="layer-detail">NO Normalization + <span class="param-activation">{disc_activation_display}</span></div>
                    {"<div class='layer-detail'>Dropout: <span class='param-dropout'>" + str(disc_dropout) + "</span></div>" if disc_dropout > 0 else ""}
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box">
                    <div class="layer-title">Layer 2: <span class="param-channels">64 → 128</span> channels</div>
                    <div class="layer-detail">Conv2d(64, 128, k=4, s=2, p=1)</div>
                    <div class="layer-detail">Output: 8×8×<span class="param-channels">128</span></div>
                    <div class="layer-detail">
                        <span class="param-norm">{disc_norm_display}</span> + 
                        <span class="param-activation">{disc_activation_display}</span>
                    </div>
                    {"<div class='layer-detail'>Dropout: <span class='param-dropout'>" + str(disc_dropout) + "</span></div>" if disc_dropout > 0 else ""}
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box">
                    <div class="layer-title">Layer 3: <span class="param-channels">128 → 256</span> channels</div>
                    <div class="layer-detail">Conv2d(128, 256, k=4, s=2, p=1)</div>
                    <div class="layer-detail">Output: 4×4×<span class="param-channels">256</span></div>
                    <div class="layer-detail">
                        <span class="param-norm">{disc_norm_display}</span> + 
                        <span class="param-activation">{disc_activation_display}</span>
                    </div>
                    {"<div class='layer-detail'>Dropout: <span class='param-dropout'>" + str(disc_dropout) + "</span></div>" if disc_dropout > 0 else ""}
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box">
                    <div class="layer-title">Final Classification Layer</div>
                    <div class="layer-detail">Conv2d(256, 1, k=4, s=1, p=0)</div>
                    <div class="layer-detail">Output: 1×1×1</div>
                    <div class="layer-detail">Sigmoid Activation</div>
                </div>
                
                <div class="arrow">↓</div>
                
                <div class="layer-box final-output">
                    Real/Fake Probability [0-1]
                </div>
            </div>
        </div>
        """
        
        return html
        
    except Exception as e:
        return f"<div style='color: red;'>Error generating diagram: {str(e)}</div>"

def load_config_with_diagram(config_path):
    """
    Load config and return the HTML diagram for it.
    Handles both simple 'best_architecture' files and complex 'nas_results' files.
    """
    if not config_path or "No configs found" in config_path:
        return "<div style='text-align: center; color: #666; padding: 20px;'>Select an architecture config to see the diagram.</div>"
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        arch_config = None
        
        # Case 1: It's a simple config file (like best_architecture_*.json)
        if 'generator' in data and 'discriminator' in data:
            arch_config = data
            
        # Case 2: It's a complex NAS results file (like nas_results_*.json)
        # We will extract and display the 'best_architecture' from it.
        elif 'best_architecture' in data and data['best_architecture']:
            arch_config = data['best_architecture']
        
        # Now, generate the diagram if we found a valid config
        if arch_config:
            return generate_architecture_diagram_html(arch_config)
        else:
            # If neither case matched, or best_architecture was null
            filename = Path(config_path).name
            return f"<div style='color: orange; padding: 20px;'>   Could not find a valid architecture definition inside <b>{filename}</b>. It might be a results file without a successful best model.</div>"

    except Exception as e:
        return f"<div style='color: red; padding: 20px;'>Error loading diagram from config: {str(e)}</div>"

#==============helpers for config path display===============#
def parse_config_info(config_path):
    """
    Extract readable strategy name from architecture config path.
    
    Examples:
        best_architecture_baseline.json -> baseline
        best_architecture_adaptive.json -> adaptive
        nas_results_random.json -> random
        architecture_G1234_D5678.json -> G1234_D5678
    
    Args:
        config_path: Path to config file
        
    Returns:
        str: Readable display name
    """
    path = Path(config_path)
    filename = path.stem  # Remove .json extension
    
    # Pattern 1: best_architecture_{strategy}.json
    match = re.search(r'best_architecture_(.+)', filename)
    if match:
        return match.group(1)
    
    # Pattern 2: nas_results_{strategy}.json
    match = re.search(r'nas_results_(.+)', filename)
    if match:
        return match.group(1)
    
    # Pattern 3: architecture_{id}.json
    match = re.search(r'architecture_(.+)', filename)
    if match:
        return match.group(1)
    
    # Fallback: return the filename without extension
    return filename


def get_architecture_configs():
    """
    Get list of available architecture configs from NAS results.
    Returns list of tuples: [(display_name, file_path), ...]
    """
    config_path = Path("experiments/gan_run_models_and_images/nas_results")
    
    if not config_path.exists():
        return [("No configs found - Run NAS first", "")]
    
    # Collect all config files
    configs = list(config_path.glob("best_architecture_*.json"))          #this json always need to contain the id so that downstream doesnot break
    # configs.extend(config_path.glob("architecture_*.json"))
    # configs.extend(config_path.glob("nas_results_*.json"))
    
    if not configs:
        return [("No configs found - Run NAS first", "")]
    
    # Sort by modification time (newest first)
    configs = sorted(configs, key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Create list of tuples: (display_name, file_path)
    config_options = []
    for config_file in configs:
        display_name = parse_config_info(config_file)
        config_options.append((display_name, str(config_file)))
    
    return config_options



# ==================== HELPER FUNCTIONS FOR VISUALIZATION ====================
def get_all_nas_results():
    """Get all NAS result files grouped by strategy"""
    results_path = Path("experiments/gan_run_models_and_images/nas_results")
    
    if not results_path.exists():
        return {}
    
    strategies = {}
    for strategy in ["adaptive", "progressive", "multifidelity", "random",'adversarial']:
        result_file = results_path / f"nas_results_{strategy}.json"
        if result_file.exists():
            strategies[strategy] = str(result_file)
    
    return strategies


def load_top_architectures(result_file, top_n=3):
    """Load top N architectures from a NAS result file"""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        if not results:
            return [], data.get('config', {}).get('strategy', 'unknown')
        
        strategy = data.get('config', {}).get('strategy', 'unknown')
        
        # Sort by enhanced_score if available, otherwise overall_score
        def get_sort_score(arch):
            return arch.get('enhanced_score', arch.get('overall_score', 0))
        
        sorted_results = sorted(results, key=get_sort_score, reverse=True)
        return sorted_results[:top_n], strategy
        
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        return [], "unknown"




def generate_score_visualization(arch_data, strategy):
    """Generate score breakdown visualization in HTML, styled to match the target."""
    arch_id = arch_data.get('architecture_id', 'Unknown')
    
    scores_html = f"""
    <style>
        .score-card {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            color: #2d3748;
            padding: 10px;
        }}
        .score-card .score-title {{ 
            font-size: 1.1em; 
            font-weight: 600; 
            border-bottom: 2px solid #cbd5e0; 
            padding-bottom: 8px; 
            margin-bottom: 15px; 
            color: #1a202c;
        }}
        .score-card .score-main {{ 
            font-size: 1.8em; 
            font-weight: 700; 
            color: #2d3748; 
            margin-bottom: 20px; 
        }}
        /* NEW RULE TO TARGET THE TEXT LABEL EXPLICITLY */
        .score-card .score-main .score-main-label {{
            color: #2d3748;
        }}
        .score-card .score-main .score-value {{
            color: #2f855a;
            background-color: #c6f6d5;
            padding: 2px 8px;
            border-radius: 6px;
        }}
        .score-card .score-category-title {{
            font-size: 0.8em;
            font-weight: 600;
            color: #718096;
            text-transform: uppercase;
            margin-top: 20px;
            margin-bottom: 5px;
        }}
        .score-card .score-item {{ 
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            padding: 8px 4px; 
            border-bottom: 1px solid #edf2f7; 
            font-size: 0.9em;
        }}
        .score-card .score-label {{ color: #4a5568; }}
        .score-card .score-value {{ 
            font-weight: 600; 
            font-family: 'Fira Code', 'Courier New', monospace; 
            color: #1a202c;
        }}
    </style>
    <div class="score-card">
        <div class="score-title">   Performance Scores</div>
    """
    
    if strategy == 'random':
        overall = arch_data.get('overall_score', 0)
        #  'score-main-label' class to the text span
        scores_html += f"<div class='score-main'><span class='score-main-label'>Overall Score: </span><span class='score-value' style='color:#2b6cb0; background-color:#bee3f8;'>{overall:.4f}</span></div>"
        
        scores_html += "<div class='score-category-title'>Component Scores</div>"
        scores_html += f"<div class='score-item'><span class='score-label'>     Stability</span> <span class='score-value'>{arch_data.get('stability_score', 0):.4f}</span></div>"
        scores_html += f"<div class='score-item'><span class='score-label'>   Loss Balance</span> <span class='score-value'>{arch_data.get('loss_balance_score', 0):.4f}</span></div>"
        scores_html += f"<div class='score-item'><span class='score-label'>      Generation Quality</span> <span class='score-value'>{arch_data.get('generation_quality_score', 0):.4f}</span></div>"
        scores_html += f"<div class='score-item'><span class='score-label'>  Training Converged</span> <span class='score-value'>{arch_data.get('training_converged', False)}</span></div>"

    else: # Enhanced scores for other strategies
        enhanced_score = arch_data.get('enhanced_score', 0)
        score_breakdown = arch_data.get('score_breakdown', {})
        
        # 'score-main-label' class to the text span
        scores_html += f"<div class='score-main'><span class='score-main-label'>    Enhanced Score: </span><span class='score-value'>{enhanced_score:.4f}</span></div>"
        
        scores_html += "<div class='score-category-title'>Detailed Score Breakdown</div>"
        scores_html += f"<div class='score-item'><span class='score-label'>  Base Score</span> <span class='score-value'>{score_breakdown.get('base', 0):.4f}</span></div>"
        scores_html += f"<div class='score-item'><span class='score-label'>  Diversity</span> <span class='score-value'>{score_breakdown.get('diversity', 0):.4f}</span></div>"
        scores_html += f"<div class='score-item'><span class='score-label'>     Class Consistency</span> <span class='score-value'>{score_breakdown.get('class_consistency', 0):.4f}</span></div>"
        scores_html += f"<div class='score-item'><span class='score-label'>   Efficiency</span> <span class='score-value'>{score_breakdown.get('efficiency', 0):.4f}</span></div>"
        scores_html += f"<div class='score-item'><span class='score-label'>🇳🇵 Nepali Quality</span> <span class='score-value'>{score_breakdown.get('nepali_quality', 0):.4f}</span></div>"

    # Common Training Metrics for all strategies
    scores_html += "<div class='score-category-title'>Training Metrics</div>"
    scores_html += f"<div class='score-item'><span class='score-label'>  Training Time</span> <span class='score-value'>{arch_data.get('training_time', 0):.2f}s</span></div>"
    scores_html += f"<div class='score-item'><span class='score-label'>  Model Size</span> <span class='score-value'>{arch_data.get('model_size_mb', 0):.2f} MB</span></div>"
    scores_html += f"<div class='score-item'><span class='score-label'>     Final G Loss</span> <span class='score-value'>{arch_data.get('final_g_loss', 0):.4f}</span></div>"
    scores_html += f"<div class='score-item'><span class='score-label'>     Final D Loss</span> <span class='score-value'>{arch_data.get('final_d_loss', 0):.4f}</span></div>"
    if 'diversity_score' in arch_data:
         scores_html += f"<div class='score-item'><span class='score-label'>   Diversity Score</span> <span class='score-value'>{arch_data.get('diversity_score', 0):.4f}</span></div>"
    scores_html += f"<div class='score-item'><span class='score-label'>  Training Converged</span> <span class='score-value'>{arch_data.get('training_converged', False)}</span></div>"
    
    scores_html += "</div>"
    return scores_html


def visualize_architecture_comparison(strategy_name):
    """Main function to visualize top 3 architectures from a strategy"""
    
    all_strategies = get_all_nas_results()
    
    if strategy_name not in all_strategies:
        return "<div style='color: red; padding: 20px; text-align:center;'>   xxx No results found for this strategy. Please run NAS first!</div>"
    
    result_file = all_strategies[strategy_name]
    top_archs, strategy = load_top_architectures(result_file, top_n=3)
    
    if not top_archs:
        return f"<div style='color: red; padding: 20px; text-align:center;'>   xxx No architectures found in {strategy_name} results!</div>"
    
    # Comprehensive HTML output
    # WRAP EVERYTHING in a main container with a defined background to prevent dark mode issues.
    output = f"""
    <style>
        /* This rule uses !important to win the CSS specificity war for the 'Sampling Strategy' text */
        .rank-card p, .rank-card strong {{
            color: #4a5568 !important; 
            margin-top: 0;
            margin-bottom: 0;
            font-size: 0.9em;
        }}
        .rank-card strong {{
            font-weight: 600 !important;
        }}
    </style>
    <div style='padding: 20px; font-family: sans-serif; background-color: white; border-radius: 12px;'>
        <h1 style='text-align: center; font-weight: 600; color: #2d3748; margin-bottom: 25px;'>    Top 3 Architectures: {strategy_name.upper()}</h1>
    """
    
    for rank, arch in enumerate(top_archs, 1):
        config = arch.get('config', {})
        arch_id = arch.get('architecture_id', 'Unknown')
        
        # 'rank-card' class for our CSS rule to target
        output += f"<div class='rank-card' style='border: 1px solid #e2e8f0; border-radius: 12px; margin-top: 30px; padding: 25px; background: #f7fafc;'>"
        output += f"<h2 style='font-size: 1.5em; color: #2d3748; margin-top:0; margin-bottom: 5px;'>  Rank {rank}: <code style='background: #edf2f7; padding: 3px 8px; border-radius: 6px; color: #4a5568;'>{arch_id}</code></h2>"
        
        # Styled the <style> block above
        sampling = config.get('sampling_strategy', 'N/A')
        output += f"<p><strong>Sampling Strategy:</strong> {sampling}</p>"
        
        
        
        # 'flex-wrap: wrap' set to 'nowrap' .
        output += "<div style='display: flex; flex-wrap: nowrap; gap: 20px; align-items: flex-start; margin-top: 20px;'>"
        
        # Architecture Diagrams
        #flex-basis that defines the initial main size of a flex item before any growing or shrinking takes place
        output += f"<div style='flex-grow: 1; flex-shrink: 0; flex-basis: 750px;'>"
        output += generate_architecture_diagram_html(config)
        output += "</div>"
        
        # Scores Card
    
        output += f"<div style='flex-grow: 1; flex-shrink: 1; min-width: 300px; background: white; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>"
        output += generate_score_visualization(arch, strategy)
        output += "</div>"
        
        output += "</div>" # Close flex container
        output += "</div>" # Close rank container
    
    output += "</div>" # Close main wrapper
    return output


# ==================== UTILITY FUNCTIONS ====================#
def extract_strategy_from_model_path(model_path):
    """
    Extract the base strategy name from a model path.
    Returns one of: 'adaptive', 'progressive', 'multifidelity', 'random', 'adversarialr None
    
    Example paths:
    - .../adaptive/models/best_generator.pth → 'adaptive'
    - .../progressive-phase-2/... → 'progressive'
    - .../multifidelity/... → 'multifidelity'
    """
    if not model_path:
        return None
    
    path_lower = str(model_path).lower()
    
    # Check for each strategy in the path
    if 'adversarial' in path_lower:
        return 'adversarial'
    elif 'adaptive' in path_lower:
        return 'adaptive'
    elif 'progressive' in path_lower:
        return 'progressive'
    elif 'multifidelity' in path_lower:
        return 'multifidelity'
    elif 'random' in path_lower:
        return 'random'
    
    return None


def get_config_for_strategy(strategy):
    """
    Get the architecture config path for a given strategy.
    
    Args:
        strategy: One of 'adaptive', 'progressive', 'multifidelity', 'random', adversarial' 
    
    Returns:
        str: Path to the config file, or None if not found
    """
    if not strategy:
        return None
    
    config_path = Path(f"experiments/gan_run_models_and_images/nas_results/best_architecture_{strategy}.json")
    
    if config_path.exists():
        return str(config_path)
    
    return None


def auto_select_config_for_model(model_path):
    """
    Automatically select the appropriate architecture config based on the model path.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        str: Path to the matching config file, or None
    """
    strategy = extract_strategy_from_model_path(model_path)
    return get_config_for_strategy(strategy)


def update_config_for_model(model_path):
    """Gradio event handler to update config dropdown when model changes."""
    auto_config = auto_select_config_for_model(model_path)
    if auto_config:
        return gr.update(value=auto_config)
    return gr.update()

def parse_model_info(model_path):
    """
    Extract strategy name, timestamp, and epoch from model path with improved formatting.
    Shows strategy, timestamp (if available), and model type.
    
    Example paths and outputs:
    - .../adaptive/models/best_generator.pth → "[Adaptive] Generator - Best"
    - .../progressive-phase-2/2025-11-06 11:27:50/models/best_generator.pth → "[Progressive Phase 2 | 2025-11-06 11:27:50] Generator - Best"
    - .../progressive-phase-2/20251106_171937/models/best_generator.pth → "[Progressive Phase 2 | 20251106_171937] Generator - Best"
    """
    path = Path(model_path)
    
    strategy = "unknown"
    timestamp = None
    
    try:
        parts = path.parts
        models_idx = parts.index('models')
        
        if 'final_training' in parts:
            training_idx = parts.index('final_training')
            if training_idx + 1 < models_idx:
                strategy = parts[training_idx + 1]
                
                if training_idx + 2 < models_idx:
                    potential_timestamp = parts[training_idx + 2]
                    
                    # --- THIS IS THE MODIFIED LINE ---
                    # Added a regex to match the 'YYYYMMDD_HHMMSS' format.
                    if (' ' in potential_timestamp or 
                        re.match(r'\d{4}-\d{2}-\d{2}', potential_timestamp) or 
                        re.match(r'\d{8}_\d{6}', potential_timestamp)):
                        timestamp = potential_timestamp
        
    except (ValueError, IndexError):
        strategy = "unknown"
    
    # Clean up strategy name for display
    strategy_display = strategy.replace('_', ' ').replace('-', ' ').title()
    
    # Build the prefix with strategy and optional timestamp
    if timestamp:
        prefix = f"[{strategy_display} | {timestamp}]"
    else:
        prefix = f"[{strategy_display}]"
    
    # Extract epoch from filename
    filename = path.stem  # filename without extension
    
    if "best_generator" in filename.lower():
        epoch_display = "Best"
    elif "generator" in filename.lower() and "epoch" in filename.lower():
        try:
            # Extract epoch number from patterns like "generator_epoch_0600"
            epoch_match = re.search(r'epoch[_-](\d+)', filename.lower())
            if epoch_match:
                epoch_num = epoch_match.group(1)
                epoch_display = f"Epoch {epoch_num}"
            else:
                epoch_display = "Unknown"
        except:
            epoch_display = "Unknown"
    else:
        epoch_display = "Unknown"
    
    # Determine if generator or discriminator
    model_type = "Generator" if "generator" in path.name.lower() else "Discriminator"
    
    return f"{prefix} {model_type} - {epoch_display}"




def get_available_models():
    """
    Get list of available trained generator models with readable names.
    Returns best_generator.pth files from:
    1. Each strategy folder (if exists directly)
    2. Each timestamped subdirectory within strategy folders
    
    This ensures all training runs are shown, not just the latest.
    """
    model_paths = []
    
    # Recursively find only BEST generator models in the final_training directory
    final_training_path = Path("experiments/gan_run_models_and_images/final_training")
    
    if final_training_path.exists():
        # Strategy 1: Look for direct best_generator.pth in strategy folders
        for strategy_dir in final_training_path.iterdir():
            if strategy_dir.is_dir():
                # Check for models/best_generator.pth directly in strategy folder
                best_gen = strategy_dir / "models" / "best_generator.pth"
                if best_gen.exists():
                    model_paths.append(best_gen)
                
                # Also check for timestamp subdirectories (like progressive-phase-2)
                for subdir in strategy_dir.iterdir():
                    if subdir.is_dir() and subdir.name not in ['models', 'images']:
                        # Check if it has a models folder with best_generator
                        best_gen_sub = subdir / "models" / "best_generator.pth"
                        if best_gen_sub.exists():
                            model_paths.append(best_gen_sub)
    
    # Also check the legacy checkpoints folder for best generators
    checkpoints = Path("experiments/gan_run_models_and_images/checkpoints")
    if checkpoints.exists():
        best_gen_checkpoint = checkpoints / "best_generator.pth"
        if best_gen_checkpoint.exists():
            model_paths.append(best_gen_checkpoint)
    
    if not model_paths:
        return [("No models found - Train a model first", "")]
    
    # Sort by modification time (newest first)
    model_paths = sorted(model_paths, key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Return list of tuples: [(label, value), ...]
    # Now we keep ALL models, including multiple from the same strategy with different timestamps
    model_options = []
    
    for p in model_paths:
        display_name = parse_model_info(p)
        model_options.append((display_name, str(p)))
    
    return model_options

def load_config_details(config_path):
    """Load and display architecture config details from simple or complex files."""
    if not config_path or "No configs found" in config_path:
        return "No configuration selected"
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        arch_config = None
        source_info = ""

        # Case 1: Simple config file
        if 'generator' in data and 'discriminator' in data:
            arch_config = data
        
        # Case 2: Complex NAS results file
        elif 'best_architecture' in data and data['best_architecture']:
            arch_config = data['best_architecture']
            source_info = f"\n\n*(This is the best architecture found in the results file `{Path(config_path).name}`)*"

        if not arch_config:
            return f"### No valid architecture found in `{Path(config_path).name}`."

        details = "### Architecture Configuration\n\n"
        details += f"**File**: {Path(config_path).name}{source_info}\n\n"
        
        if 'generator' in arch_config:
            details += "**Generator**:\n"
            for key, value in arch_config['generator'].items():
                details += f"  - `{key}`: {value}\n"
            details += "\n"
        
        if 'discriminator' in arch_config:
            details += "**Discriminator**:\n"
            for key, value in arch_config['discriminator'].items():
                details += f"  - `{key}`: {value}\n"
            details += "\n"
        
        # 'score' is usually in the simple file, let's check for it
        if 'score' in arch_config:
            details += f"**NAS Score**: {arch_config['score']:.4f}\n"
        
        return details

    except Exception as e:
        return f"Error loading config: {str(e)}"



def generate_digits(model_path, arch_config_path, digit_class, num_samples, grid_layout):
    """Generate Devanagari digits using trained model"""

    if "No models found" in model_path or "No configs found" in arch_config_path:
        return None, "   xxx Please train a model and run NAS first!"

    try:
        # Create output directory
        output_dir = Path("experiments/gan_run_models_and_images/gradio_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run the generator script
        cmd = [
            "python", "scripts/generate.py",
            "--model", model_path,
            "--arch-config", arch_config_path,
            "--class", str(digit_class),
            "--num-samples", str(num_samples),
            "--output-dir", str(output_dir)
        ]
        if grid_layout:
            cmd.append("--grid")

        
        # ---------------------------------------------------#
        # DELETE OLD IMAGES for this digit_class + architecture #
        # ---------------------------------------------------#
        arch_id = None
        try:
            with open(arch_config_path, 'r') as f:
                arch_data = json.load(f)
                arch_id = f"{arch_data.get('id', 'unknown')}_{arch_data.get('sampling_strategy', 'random')}"
        except Exception as e:
            print(f"   Could not extract architecture info: {e}")
            arch_id = "unknown_arch"

        # Target pattern (e.g. class_0_G9596_D8925_random)
        class_dir_pattern = f"class_{digit_class}_{arch_id}*"

        # Delete only those old PNGs
        for d in output_dir.glob(class_dir_pattern):
            print(f"  Clearing old samples in: {d}")
            for p in d.glob("*.png"):
                try:
                    p.unlink()
                except Exception as e:
                    print(f"   Could not delete {p}: {e}")
        
        
        
        
        subprocess.run(cmd, check=True)

        # After generation, show only the *latest* generated outputs
        if grid_layout:
            # Find only the newest grid image
            gen_files = list(output_dir.glob(f"generated_grid_class_{digit_class}_*.png"))
            if not gen_files:
                gen_files = list(output_dir.glob("generated_grid_*.png"))
            if gen_files:
                img_path = max(gen_files, key=os.path.getmtime)
                img = Image.open(img_path)
                return [img], f"  Generated grid for digit {digit_class}\n📁 {img_path}"
            else:
                return None, "   xxx No grid image found after generation!"
        else:
            # Find the most recent class folder created
            class_folders = list(output_dir.glob(class_dir_pattern))
            if not class_folders:
                return None, f"   xxx No folder found for class {digit_class}!"
            latest_class_dir = max(class_folders, key=os.path.getmtime)

            # Only load the images from this folder
            indiv_files = sorted(latest_class_dir.glob("*.png"), key=os.path.getmtime)
            images = [Image.open(f) for f in indiv_files]

            return images, f"  Generated {len(images)} samples for digit {digit_class}\n📁 {latest_class_dir}"

    except Exception as e:
        return None, f"   xxx Error: {str(e)}"


def run_nas_experiment(strategy, num_architectures, epochs_per_arch, max_subset, split, progress=gr.Progress()):
    """Run NAS experiment with selected strategy"""
    
    try:
        progress(0, desc="Initializing NAS experiment...")
        
        # Build command
        cmd = [
            "python", "-u", "scripts/nas_search.py",
            "--strategy", strategy,
            "--num-archs", str(num_architectures),
            "--epochs", str(epochs_per_arch)
        ]
        
        # --- LOGIC TO ADD NEW ARGUMENTS TO THE CMD ---
        if max_subset is not None:
            cmd.extend(["--max-subset-per-class", str(int(max_subset))])
        
        if split is not None:
            cmd.extend(["--split", str(split)])
        
        
        
        # Run NAS
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for i, line in enumerate(process.stdout):
            output_lines.append(line)
            
            # Update progress
            if "Architecture" in line and "/" in line:
                try:
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if "/" in part:
                            current, total = part.split("/")
                            current = int(current.strip("[]"))
                            total = int(total.strip("[]"))
                            progress_pct = current / total
                            progress(progress_pct, desc=f"Evaluating architecture {current}/{total}")
                            break
                except:
                    pass
            
            # Yield output every 5 lines for responsiveness
            if i % 5 == 0:
                yield "\n".join(output_lines[-100:])  # Show last 100 lines
        
        process.wait()
        
        if process.returncode == 0:
            final_output = "\n".join(output_lines)
            final_output += "\n\n  NAS experiment completed successfully!"
            final_output += f"\n   Results saved to: experiments/gan_run_models_and_images/nas_results/"
            yield final_output
        else:
            yield "\n".join(output_lines) + "\n\n   xxx NAS experiment failed!"
            
    except Exception as e:
        yield f"   xxx Error running NAS: {str(e)}" 
        

def train_model_from_scratch(arch_config_path, epochs, batch_size, g_lr, d_lr,max_subset, split, progress=gr.Progress()):
    """Train a new model from scratch using architecture config"""
    
    if "No configs found" in arch_config_path:
        yield "   xxx Please run NAS first to get architecture configurations!"
        return
    
    try:
        progress(0, desc="Initializing training...")
        
        # Build command
        cmd = [
            "python", "scripts/train_model.py",
            "--arch-config", arch_config_path,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--g-lr", str(g_lr),
            "--d-lr", str(d_lr)
        ]
        
        if max_subset is not None:
            cmd.extend(["--max-subset-per-class", str(int(max_subset))])
        
        if split is not None:
            cmd.extend(["--split", str(split)])
        
        cmd = ["python", "-u"] + cmd[1:]  # <- the -u flag forces unbuffered output
        
        # Run training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for i, line in enumerate(process.stdout):
            output_lines.append(line)
            
            # Update progress based on epoch
            if "Epoch" in line:
                try:
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if part == "Epoch":
                            epoch_info = parts[j+1].strip("[]")
                            if "/" in epoch_info:
                                current, total = epoch_info.split("/")
                                current = int(current)
                                total = int(total)
                                progress_pct = current / total
                                progress(progress_pct, desc=f"Training: Epoch {current}/{total}")
                except:
                    pass
            
            # Yield output periodically
            if i % 5 == 0 or "Epoch" in line or "Loss" in line:
                yield "\n".join(output_lines[-50:])
        
        process.wait()
        
        if process.returncode == 0:
            final_output = "\n".join(output_lines)
            final_output += "\n\n  Training completed successfully!"
            final_output += f"\n  Models saved to: experiments/gan_run_models_and_images/checkpoints/"
            yield final_output
        else:
            yield "\n".join(output_lines) + "\n\n   xxx Training failed!"
            
    except Exception as e:
        yield f"   xxx Error during training: {str(e)}" 
        
def resume_training(checkpoint_path, arch_config_path, additional_epochs, g_lr, d_lr,max_subset, split,  progress=gr.Progress()):
    """Resume training from a checkpoint"""
    
    if "No models found" in checkpoint_path:
        yield "   xxx No checkpoints available to resume from!"
        return
    
    if "No configs found" in arch_config_path:
        yield "   xxx Please select an architecture config!"
        return
    
    try:
        progress(0, desc="Resuming training from checkpoint...")
        
        # Build command
        cmd = [
            "python","-u", "scripts/train_model.py",
            "--arch-config", arch_config_path,
            "--resume-from", checkpoint_path,
            "--epochs", str(additional_epochs),
            "--g-lr", str(g_lr),
            "--d-lr", str(d_lr)
        ]
        
        
        if max_subset is not None:
            cmd.extend(["--max-subset-per-class", str(int(max_subset))])
        
        if split is not None:
            cmd.extend(["--split", str(split)])
        
        # Run training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for i, line in enumerate(process.stdout):
            output_lines.append(line)
            
            # Update progress
            if "Epoch" in line:
                try:
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if part == "Epoch":
                            epoch_info = parts[j+1].strip("[]")
                            if "/" in epoch_info:
                                current, total = epoch_info.split("/")
                                progress_pct = int(current) / int(total)
                                progress(progress_pct, desc=f"Resuming: Epoch {current}/{total}")
                except:
                    pass
            
            if i % 3 == 0:
                yield "\n".join(output_lines[-50:])
        
        process.wait()
        
        if process.returncode == 0:
            final_output = "\n".join(output_lines)
            final_output += "\n\n  Continued training completed!"
            yield final_output
        else:
            yield "\n".join(output_lines) + "\n\n   xxx Training continuation failed!"
            
    except Exception as e:
        yield f"   xxx Error resuming training: {str(e)}"  
        



def view_nas_results(results_file):
    """Display NAS experiment results"""
    
    if not results_file or len(results_file) == 0:
        return "No NAS results available. Run a NAS experiment first!"
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        output = f"# NAS Results: {Path(results_file).name}\n\n"
        
        #  Use .get() to safely handle cases where best_architecture is null/None 
        best_arch = results.get('best_architecture')
        
        # Display Best Architecture 
        if best_arch: # This check handles both a missing key and a null value
            output += "##     Best Architecture\n"
            
            best_score_val = "N/A"
            if 'summary' in results and 'best_score' in results['summary']:
                best_score_val = f"{results['summary']['best_score']:.4f}"
            
            output += f"**Score**: {best_score_val}\n\n"
            
            if 'generator' in best_arch:
                output += "**Generator Config**:\n"
                for k, v in best_arch['generator'].items():
                    output += f"  - {k}: {v}\n"
                output += "\n"
            
            if 'discriminator' in best_arch:
                output += "**Discriminator Config**:\n"
                for k, v in best_arch['discriminator'].items():
                    output += f"  - {k}: {v}\n"
                output += "\n"
        else:
            # Handle the case where no best architecture was found
            output += "##     Best Architecture\n"
            output += "**No successful architecture was found in this NAS run.**\n\n"

        #  Display Ranked List of All Architectures 
        all_results_list = results.get('results')
        
        if all_results_list:
            output += f"##    All Architectures ({len(all_results_list)} total)\n\n"
            
            def get_score_for_ranking(arch):
                if 'enhanced_score' in arch:
                    return arch['enhanced_score']
                if 'overall_score' in arch:
                    return arch['overall_score']
                return 0
            
            sorted_results = sorted(
                all_results_list,
                key=get_score_for_ranking,
                reverse=True
            )
            
            for i, arch in enumerate(sorted_results[:10], 1):
                score_value = get_score_for_ranking(arch)
                arch_id = arch.get('architecture_id', 'Unknown ID')

                if score_value > 0:
                    output += f"**Rank {i}**: `{arch_id}` - Score = {score_value:.4f}\n"
                else: 
                    output += f"**Rank {i}**: `{arch_id}` - Score = N/A\n"
        return output
        
    except Exception as e:
        # str(e) shows the original error for debugging
        return f"Error loading results: {str(e)}"
    
    
#=================helpers for model evaluation ===============#
def get_evaluation_results():
    """Get all evaluation result files."""
    eval_path = Path("experiments/evaluation")
    
    if not eval_path.exists():
        return []
    
    results = list(eval_path.glob("*.json"))
    return sorted([str(r) for r in results], key=os.path.getmtime, reverse=True)



def run_model_evaluation(gen_model, disc_model, arch_config, use_enhanced, progress=gr.Progress()):
    """Run evaluation on a trained model - FIXED VERSION."""
    
    if "No models found" in gen_model:
        return "   xxx Please select a valid generator model!"
    
    try:
        progress(0.2, desc="Preparing evaluation...")
        
        # Build command
        cmd = [
            "python", "scripts/evaluate_model.py",
            "--generator", gen_model,
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ]
        
        # Add discriminator if provided
        if disc_model and "No models found" not in disc_model:
            cmd.extend(["--discriminator", disc_model])
        # Otherwise, let the script auto-detect it
        
        # Add arch config if provided
        if arch_config and "No configs found" not in arch_config:
            cmd.extend(["--arch-config", arch_config])
        # Otherwise, let the script auto-detect based on path
        
        if use_enhanced:
            cmd.append("--use-enhanced")
        
        progress(0.4, desc="Running evaluation...")
        
        # Run evaluation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        progress(1.0, desc="Evaluation complete!")
        
        output = result.stdout
        if result.returncode != 0:
            output += f"\n\n   xxx Evaluation failed!\n{result.stderr}"
            return output
        
        if result.stderr:
            output += "\n\nWarnings:\n" + result.stderr
        
        return output + "\n\n  Evaluation completed successfully!"
        
    except subprocess.TimeoutExpired:
        return "   xxx Evaluation timed out (exceeded 5 minutes)"
    except Exception as e:
        return f"   xxx Error: {str(e)}"


def run_batch_evaluation(eval_dir, use_enhanced, progress=gr.Progress()):
    """Run batch evaluation on all models in directory - FIXED VERSION."""
    
    if not eval_dir or not Path(eval_dir).exists():
        return f"   xxx Directory not found: {eval_dir}"
    
    try:
        progress(0.1, desc="Scanning directory...")
        
        cmd = [
            "python", "scripts/evaluate_model.py",
            "--eval-dir", eval_dir,
            "--recursive",
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ]
        
        if use_enhanced:
            cmd.append("--use-enhanced")
        
        progress(0.3, desc="Evaluating models...")
        
        # Run batch evaluation with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout for batch
        )
        
        progress(1.0, desc="Batch evaluation complete!")
        
        output = result.stdout
        if result.returncode != 0:
            output += f"\n\n   xxx Batch evaluation failed!\n{result.stderr}"
            return output
        
        if result.stderr:
            output += "\n\nWarnings:\n" + result.stderr
        
        return output + "\n\n  Batch evaluation completed!"
        
    except subprocess.TimeoutExpired:
        return "   xxx Batch evaluation timed out (exceeded 30 minutes)"
    except Exception as e:
        return f"   xxx Error: {str(e)}"


def visualize_evaluation_results(results_file):
    """Visualize evaluation results with enhanced formatting - SHOWS ARCHITECTURE ID AND STRATEGY."""
    
    if not results_file or len(results_file) == 0:
        return "<div style='text-align: center; color: #666; padding: 20px;'>No evaluation results available. Run an evaluation first!</div>"
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        is_batch = 'results' in data and isinstance(data['results'], list)
        
        output = f"""
        <div style='font-family: sans-serif; padding: 20px; background: #f8f9fa; border-radius: 12px;'>
            <h1 style='color: #2d3748; margin-bottom: 20px;'>   Evaluation Results</h1>
        """
        
        if is_batch:
            output += f"""
            <div style='background: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e8f0;'>
                <h3 style='color: #2d3748; margin-top: 0;'>Batch Evaluation Summary</h3>
                <p style='color: #4a5568; margin: 8px 0;'><strong style='color: #2d3748;'>Timestamp:</strong> {data.get('timestamp', 'N/A')}</p>
                <p style='color: #4a5568; margin: 8px 0;'><strong style='color: #2d3748;'>Number of Models:</strong> {data.get('num_models', 0)}</p>
                <p style='color: #4a5568; margin: 8px 0;'><strong style='color: #2d3748;'>Evaluation Type:</strong> {'Enhanced' if data.get('enhanced_evaluation') else 'Standard'}</p>
            </div>
            """
            
            results = sorted(
                data['results'],
                key=lambda x: x.get('enhanced_score', x.get('overall_score', 0)),
                reverse=True
            )
            
            for i, result in enumerate(results, 1):
                model_info = result.get('model_info', {})
                arch_id = model_info.get('architecture_id', 'Unknown')
                sampling_strategy = model_info.get('sampling_strategy', 'Unknown')
                strategy = result.get('strategy', 'Unknown')
                score = result.get('enhanced_score', result.get('overall_score', 0))
                
                output += f"""
                <div style='border: 1px solid #e2e8f0; padding: 20px; margin: 20px 0; border-radius: 8px; background: #ffffff;'>
                    <h2 style='color: #2d3748; margin-top: 0;'>#{i} - {strategy}</h2>
                    <div style='background: #f7fafc; padding: 12px; border-radius: 6px; margin: 10px 0; border: 1px solid #e2e8f0;'>
                        <p style='margin: 8px 0;'><strong style='color: #1a202c;'>    Architecture ID:</strong> <code style='background: #edf2f7; padding: 4px 8px; border-radius: 4px; color: #1a202c; border: 1px solid #cbd5e0;'>{arch_id}</code></p>
                        <p style='margin: 8px 0;'><strong style='color: #1a202c;'>     Sampling Strategy:</strong> <code style='background: #edf2f7; padding: 4px 8px; border-radius: 4px; color: #1a202c; border: 1px solid #cbd5e0;'>{sampling_strategy}</code></p>
                    </div>
                """
                
                if 'enhanced_score' in result:
                    output += f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 15px; border-radius: 8px; margin: 15px 0;'>
                        <div style='font-size: 1.8em; font-weight: bold;'>
                                Enhanced Score: {result['enhanced_score']:.4f}
                        </div>
                    </div>
                    """
                    
                    if 'score_breakdown' in result:
                        output += "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 15px 0;'>"
                        for key, value in result['score_breakdown'].items():
                            output += f"""
                            <div style='background: #f7fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0;'>
                                <div style='color: #4a5568; font-size: 0.9em;'>{key.replace('_', ' ').title()}</div>
                                <div style='color: #1a202c; font-size: 1.3em; font-weight: 600;'>{value:.4f}</div>
                            </div>
                            """
                        output += "</div>"
                else:
                    output += f"""
                    <div style='font-size: 1.5em; color: #2f855a; font-weight: bold; margin: 10px 0;'>
                        Overall Score: {score:.4f}
                    </div>
                    """
                
                if model_info:
                    output += f"""
                    <div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;'>
                        <h4 style='color: #2d3748; margin-top: 0;'>Model Information</h4>
                        <p style='margin: 5px 0;'><strong style='color: #1a202c;'>Total Size:</strong> <span style='color: #2d3748;'>{model_info.get('total_size_mb', 0):.2f} MB</span></p>
                        <p style='margin: 5px 0;'><strong style='color: #1a202c;'>Generator:</strong> <span style='color: #2d3748;'>{model_info.get('generator_size_mb', 0):.2f} MB</span></p>
                        <p style='margin: 5px 0;'><strong style='color: #1a202c;'>Discriminator:</strong> <span style='color: #2d3748;'>{model_info.get('discriminator_size_mb', 0):.2f} MB</span></p>
                    </div>
                    """
                
                output += "</div>"
        
        else:
            # Single model evaluation
            model_info = data.get('model_info', {})
            arch_id = model_info.get('architecture_id', 'Unknown')
            sampling_strategy = model_info.get('sampling_strategy', 'Unknown')
            
            output += f"""
            <div style='background: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e8f0;'>
                <h3 style='color: #2d3748; margin-top: 0;'>Single Model Evaluation</h3>
                <div style='margin-top: 10px;'>
                    <p style='margin: 8px 0;'><strong style='color: #1a202c;'>    Architecture ID:</strong> <code style='background: #f7fafc; padding: 4px 8px; border-radius: 4px; color: #1a202c; border: 1px solid #e2e8f0;'>{arch_id}</code></p>
                    <p style='margin: 8px 0;'><strong style='color: #1a202c;'>     Sampling Strategy:</strong> <code style='background: #f7fafc; padding: 4px 8px; border-radius: 4px; color: #1a202c; border: 1px solid #e2e8f0;'>{sampling_strategy}</code></p>
                </div>
            </div>
            """
            
            if 'enhanced_score' in data:
                output += f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 20px; border-radius: 8px; margin: 20px 0;'>
                    <div style='font-size: 2em; font-weight: bold;'>
                            Enhanced Score: {data['enhanced_score']:.4f}
                    </div>
                </div>
                """
                
                if 'score_breakdown' in data:
                    output += "<h3 style='color: #2d3748; margin-top: 20px;'>Score Breakdown</h3>"
                    output += "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;'>"
                    for key, value in data['score_breakdown'].items():
                        output += f"""
                        <div style='background: #f7fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0;'>
                            <div style='color: #4a5568; font-size: 1em; margin-bottom: 5px;'>{key.replace('_', ' ').title()}</div>
                            <div style='color: #1a202c; font-size: 1.5em; font-weight: 600;'>{value:.4f}</div>
                        </div>
                        """
                    output += "</div>"
            
            if model_info:
                output += f"""
                <div style='background: #ffffff; padding: 20px; border-radius: 8px; margin-top: 20px; border: 1px solid #e2e8f0;'>
                    <h3 style='color: #2d3748; margin-top: 0;'>Model Information</h3>
                    <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
                        <div>
                            <p style='margin: 5px 0;'><strong style='color: #1a202c;'>Total Size:</strong> <span style='color: #2d3748;'>{model_info.get('total_size_mb', 0):.2f} MB</span></p>
                        </div>
                        <div>
                            <p style='margin: 5px 0;'><strong style='color: #1a202c;'>Generator:</strong> <span style='color: #2d3748;'>{model_info.get('generator_size_mb', 0):.2f} MB</span></p>
                            <p style='margin: 5px 0;'><strong style='color: #1a202c;'>Discriminator:</strong> <span style='color: #2d3748;'>{model_info.get('discriminator_size_mb', 0):.2f} MB</span></p>
                        </div>
                    </div>
                </div>
                """
        
        output += "</div>"
        return output
        
    except Exception as e:
        return f"<div style='color: red; padding: 20px;'>Error loading results: {str(e)}</div>"


# ==================== GRADIO INTERFACE ====================#

def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(
        title="NepScript Genesis - Devanagari GAN",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        #       NepScript Genesis
        ### Neural Architecture Search for Devanagari Digit Generation
        
        Generate synthetic Nepali/Devanagari handwritten digits (०-९) using Conditional GANs
        """)
        
        # ==================== TAB 1: GENERATE DIGITS ====================#
        with gr.Tab("     Generate Digits"):
            gr.Markdown("### Generate Devanagari digits using trained models")
            
            with gr.Row():
                with gr.Column(scale=1):
                    models = get_available_models()
                    default_model = models[0][1] if models and models[0][0] != "No models found - Train a model first" else None
                    
                    gen_model = gr.Dropdown(
                        choices=models,
                        label="Select Trained Model",
                        value=default_model,
                        interactive=True
                    )
                    
                    configs = get_architecture_configs()
                    # Auto-select config based on default model
                    default_config = auto_select_config_for_model(default_model) if default_model else None
                    if not default_config and configs and configs[0][0] != "No configs found - Run NAS first":
                        default_config = configs[0][1]
                    
                    gen_config = gr.Dropdown(
                            choices=configs,
                            label="Architecture Config (Auto-selected)",
                            value=default_config,
                            interactive=True
                    )
                    
                    
                    digit_class = gr.Slider(
                        minimum=0,
                        maximum=9,
                        step=1,
                        value=0,
                        label="Digit Class (०-९)",
                        info="Which Devanagari digit to generate"
                    )
                    
                    num_samples = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=16,
                        label="Number of Samples"
                    )
                    
                    grid_layout = gr.Checkbox(
                        value=True,
                        label="Grid Layout",
                        info="Arrange samples in a grid"
                    )
                    
                    generate_btn = gr.Button("      Generate", variant="primary", size="lg")
                    with gr.Row():
                        refresh_models_btn = gr.Button("  Refresh Models List", size="sm")
                        refresh_gen_configs_btn = gr.Button("  Refresh Configs List", size="sm")
                    # refresh_models_btn = gr.Button("  Refresh Models List", size="sm")
                
                with gr.Column(scale=2):
                    generated_gallery = gr.Gallery(
                        label="Generated Samples",
                        show_label=True,
                        columns=4,
                        height="auto"
                    )
                    generation_status = gr.Textbox(label="Status", lines=3)
            
            
            # text details and visual diagram side by side in a row 
            with gr.Row(equal_height= True):
                with gr.Column(scale = 1, min_width = 350):
                    config_details = gr.Markdown("Select a config to view details")
                
                with gr.Column(scale =2):
                    arch_diagram_display = gr.HTML(label = "Architecture Diagram")
            # Event handlers
            generate_btn.click(
                fn=generate_digits,
                inputs=[gen_model, gen_config, digit_class, num_samples, grid_layout],
                outputs=[generated_gallery, generation_status]
            )
            
            # Auto-select config when model changes
            gen_model.change(
                fn=update_config_for_model,
                inputs=[gen_model],
                outputs=[gen_config]
            )

            
            refresh_models_btn.click(
                fn=lambda: gr.update(choices=get_available_models()),
                outputs=gen_model
            )
            
            refresh_gen_configs_btn.click(
                fn=lambda: gr.update(choices=get_architecture_configs()),
                outputs=gen_config
            )
            
            gen_config.change(
                fn=load_config_details,
                inputs=gen_config,
                outputs=config_details
            ) 
            # HANDLER: updates the VISUAL diagram
            gen_config.change( 
                fn=load_config_with_diagram,
                inputs=gen_config,
                outputs=arch_diagram_display
            )
            
            # Trigger diagram generation on app load for the default selection
            app.load( 
                fn=load_config_with_diagram,
                inputs=gen_config,
                outputs=arch_diagram_display
            )

        #===============COMPARE MODELS TAB ======================#
        with gr.Tab("  Compare Models"):
            gr.Markdown("### Dynamically compare 2–4 models side-by-side")

            with gr.Row():
                with gr.Column():
                    models = get_available_models()
                    config_choices = get_architecture_configs()

                    # Slider to select how many models to compare
                    num_models = gr.Slider(
                        minimum=2,
                        maximum=4,
                        step=1,
                        value=2,
                        label="Number of Models to Compare"
                    )

                    # --- Placeholders for dynamic components ---
                    model_dropdowns = []
                    config_dropdowns = []

                    # Create maximum possible components ( Only hide/show dynamically)

                    for i in range(4):
                        # Get default model value for first 2 slots
                        default_model = None
                        if i < 2 and models and models[0][0] != "No models found - Train a model first":
                            if i < len(models):
                                default_model = models[i][1]
                        
                        model_dropdowns.append(
                            gr.Dropdown(
                                choices=models,
                                label=f"Model {chr(65 + i)}",  # A, B, C, D
                                visible=(i < 2),  # only show 2 initially
                                value=default_model
                            )
                        )
                        
                        # Auto-select config based on model
                        default_config = auto_select_config_for_model(default_model) if default_model else None
                        
                        config_dropdowns.append(
                            gr.Dropdown(
                                choices=config_choices,
                                label=f"Config {chr(65 + i)} (Auto-selected)",
                                visible=(i < 2),
                                value=default_config
                            )
                        )

                    # ---  Checkbox for generating all classes ---#
                    compare_all_classes = gr.Checkbox(
                        label="Generate for All 10 Classes",
                        value=False,
                        info="If checked, the digit class slider will be ignored."
                    )
                    
                    compare_digit = gr.Slider(0, 9, step=1, value=0, label="Digit Class (if not generating all)")
                    compare_samples = gr.Slider(1, 4, step=1, value=2, label="Samples per Class", info="No of samples to generate per class for each model")
                    compare_btn = gr.Button("   Compare", variant="primary")
                    with gr.Row():
                        refresh_compare_models_btn = gr.Button("  Refresh Models List", size="sm")
                        refresh_compare_configs_btn = gr.Button("  Refresh Configs List", size="sm")
                with gr.Column():
                    # Create galleries for up to 4 models
                    galleries = []
                    for i in range(4):
                        galleries.append(gr.Gallery(label=f"Model {chr(65 + i)}", columns=8, visible=(i < 2), height="auto"))

                    comparison_status = gr.Textbox(label="Status", lines=3)

            # --- Function to update visible components ---#
            def update_visible_models(n_models):
                # Create separate lists for each component type
                model_updates = []
                config_updates = []
                gallery_updates = []

                for i in range(4):
                    is_visible = i < n_models
                    model_updates.append(gr.update(visible=is_visible))
                    config_updates.append(gr.update(visible=is_visible))
                    gallery_updates.append(gr.update(visible=is_visible))

                # Concatenate the lists in the SAME order as the `outputs` parameter
                return model_updates + config_updates + gallery_updates
            
            num_models.change(
                fn=update_visible_models,
                inputs=[num_models],
                outputs=model_dropdowns + config_dropdowns + galleries
            )
            def refresh_all_model_dropdowns():
                new_choices = get_available_models()
                return [gr.update(choices=new_choices) for _ in range(4)]

            def refresh_all_config_dropdowns():
                new_choices = get_architecture_configs()
                return [gr.update(choices=new_choices) for _ in range(4)]
            
            for i, model_dropdown in enumerate(model_dropdowns):
                model_dropdown.change(
                    fn=update_config_for_model,
                    inputs=[model_dropdown],
                    outputs=[config_dropdowns[i]]
                )

            
            # ---Comparison function to handle all classes ---#
            def compare_models_dynamic(digit, samples, generate_all, *args):
                # Unpack models and configs
                models = args[:4]
                configs = args[4:]

                results = []
                active_models_status = []
                
                # Determine which classes to generate
                classes_to_generate = range(10) if generate_all else [digit]
                num_classes = len(classes_to_generate)

                # Loop through each of the 4 model slots
                for i in range(4):
                    model_name = models[i]
                    config_name = configs[i]
                    
                    # This list will hold (image, caption) tuples for the current model
                    model_images_with_captions = []

                    if model_name and config_name:
                        print(f"  Processing Model {chr(65 + i)} for {num_classes} class(es)...")
                        try:
                            # Loop through each required digit class
                            for class_digit in classes_to_generate:
                                # Call your existing generation function for each class
                                generated_images, _ = generate_digits(model_name, config_name, class_digit, int(samples), False)
                                
                                # Add images with captions to the list
                                if generated_images:
                                    for img in generated_images:
                                        model_images_with_captions.append((img, f"Digit {class_digit}"))
                            
                            # If we successfully generated images, add the list to results
                            if model_images_with_captions:
                                results.append(model_images_with_captions)
                                model_label = Path(model_name).stem
                                active_models_status.append(f"Model {chr(65 + i)} ({model_label})")
                            else:
                                results.append(None) # Handle case where generation produced nothing
                        
                        except Exception as e:
                            results.append(None) # Append a placeholder for failed generation
                            print(f"   xxx Error generating images for Model {chr(65 + i)}: {e}")
                    else:
                        # This slot is not active, so append None
                        results.append(None)
                
                # Build the final status message
                if any(active_models_status):
                    class_info = f"all {num_classes} classes" if generate_all else f"digit {digit}"
                    status = f"  Generated {int(samples)} samples for {class_info} from: {', '.join(active_models_status)}"
                else:
                    status = "   No models were selected or able to generate images. Check console for errors."

                # Return a value for each of the 4 galleries and the status textbox
                return *results, status
            
            # --- Event handlers for refresh buttons --- #
            refresh_compare_models_btn.click(
                fn=refresh_all_model_dropdowns,
                inputs=None,
                outputs=model_dropdowns
            )

            refresh_compare_configs_btn.click(
                fn=refresh_all_config_dropdowns,
                inputs=None,
                outputs=config_dropdowns
            )
            
            compare_btn.click(
                fn=compare_models_dynamic,
                # Add the new checkbox to the inputs list
                inputs=[compare_digit, compare_samples, compare_all_classes] + model_dropdowns + config_dropdowns,
                outputs=galleries + [comparison_status]
            )
        #=========================Compare Architecture===================#
        with gr.Tab("    Compare Architectures"):
            gr.Markdown('''
            ### Compare Top 3 Architectures from Each NAS Strategy
            
            Visualize and compare the best architectures discovered by different search strategies.
            See detailed architecture diagrams, hyperparameters, and performance scores.
            ''')
            
            with gr.Row():
                strategy_selector = gr.Radio(
                    choices=["adaptive", "progressive", "multifidelity", "random"],
                    value="adaptive",
                    label="Select NAS Strategy",
                    info="Choose which strategy's results to visualize"
                )
            
            compare_arch_btn = gr.Button("    Visualize Top 3 Architectures", variant="primary", size="lg")
            

            architecture_display = gr.HTML(
                value="<div style='text-align: center; color: #666; padding: 20px;'>Select a strategy and click 'Visualize' to see the top architectures</div>",
                label="Architecture Comparison"
            )
            
            
            
            compare_arch_btn.click(
                fn=visualize_architecture_comparison,
                inputs=strategy_selector,
                outputs=architecture_display
            )
            
            
#==================== EVALUATE MODELS TAB ====================#


        with gr.Tab("     Evaluate Models"):
            gr.Markdown("""
            ### Evaluate Trained Models
            
            Evaluate your trained models using the same comprehensive metrics that NAS uses during architecture search.
            Supports single model evaluation or batch evaluation of all models.
            """)
            
            with gr.Tabs():
                # Single Model Evaluation
                with gr.Tab("   Single Model"):
                    with gr.Row():
                        with gr.Column():
                            models = get_available_models()
                            eval_gen_model = gr.Dropdown(
                                choices=models,
                                label="Generator Model",
                                value=models[0][1] if models and models[0][0] != "No models found - Train a model first" else None
                            )
                            
                            eval_disc_model = gr.Dropdown(
                                choices=models,
                                label="Discriminator Model (optional - will auto-detect)",
                                value=None
                            )
                            
                            configs = get_architecture_configs()
                            
                            default_eval_model = models[0][1] if models and models[0][0] != "No models found - Train a model first" else None
                            default_eval_config = auto_select_config_for_model(default_eval_model) if default_eval_model else None
                            
                            eval_arch_config = gr.Dropdown(
                                choices=configs,
                                label="Architecture Config",
                                value= default_eval_config,
                            )
                            
                            eval_use_enhanced = gr.Checkbox(
                                value=True,
                                label="Use Enhanced Evaluation",
                                info="Uses ensemble evaluator with multiple metrics"
                            )
                            
                            eval_single_btn = gr.Button("     Evaluate Model", variant="primary", size="lg")
                            
                            with gr.Row():
                                refresh_eval_models_btn = gr.Button("  Refresh Models", size="sm")
                                refresh_eval_configs_btn = gr.Button("  Refresh Configs", size="sm")
                        
                        with gr.Column():
                            eval_output = gr.Textbox(
                                label="Evaluation Output",
                                lines=20,
                                # show_copy_button=True
                            )
                    
                    eval_single_btn.click(
                        fn=run_model_evaluation,
                        inputs=[eval_gen_model, eval_disc_model, eval_arch_config, eval_use_enhanced],
                        outputs=eval_output
                    )
                    eval_gen_model.change(
                        fn=update_config_for_model,
                        inputs=[eval_gen_model],
                        outputs=[eval_arch_config]
                    )
                    refresh_eval_models_btn.click(
                        fn=lambda: [gr.update(choices=get_available_models()), gr.update(choices=get_available_models())],
                        outputs=[eval_gen_model, eval_disc_model]
                    )
                    
                    refresh_eval_configs_btn.click(
                        fn=lambda: gr.update(choices=get_architecture_configs()),
                        outputs=eval_arch_config
                    )
            # View Results Section
            gr.Markdown("---")
            gr.Markdown("###    View Evaluation Results")
            
            with gr.Row():
                eval_results_dropdown = gr.Dropdown(
                    choices=get_evaluation_results(),
                    label="Select Evaluation Results",
                    value=get_evaluation_results()[0] if get_evaluation_results() else None
                )
                
                with gr.Row():
                    view_eval_btn = gr.Button("👁️ View Results", variant="primary")
                    refresh_eval_results_btn = gr.Button("  Refresh Results", size="sm")
            
            eval_results_display = gr.HTML(
                value="<div style='text-align: center; color: #666; padding: 20px;'>Select results and click 'View Results'</div>"
            )
            
            view_eval_btn.click(
                fn=visualize_evaluation_results,
                inputs=eval_results_dropdown,
                outputs=eval_results_display
            )
            
            refresh_eval_results_btn.click(
                fn=lambda: gr.update(choices=get_evaluation_results()),
                outputs=eval_results_dropdown
            )
            
    # ==================== LOAD HANDLERS (Trigger on app start) ====================#
    # Trigger config details and diagram display on initial load for Generate Digits tab
        app.load(
            fn=load_config_details,
            inputs=[gen_config],
            outputs=[config_details]
        )
        
        app.load(
            fn=load_config_with_diagram,
            inputs=[gen_config],
            outputs=[arch_diagram_display]
        )
        
            
            
    return app



# ==================== MAIN ====================#

if __name__ == "__main__":
    app = create_interface()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
