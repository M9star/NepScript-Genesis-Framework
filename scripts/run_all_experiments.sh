#!/bin/bash

# NepScript Genesis - Batch Experiment Script
# This script runs the full pipeline (NAS + Final Training) for all supported strategies.
# Note: This may take several hours depending on your hardware.

set -e # Exit on error

# Define a stable output directory for results
NAS_ROOT="experiments/gan_run_models_and_images/nas_results"
mkdir -p "$NAS_ROOT"

echo "========================================================="
echo "STARTING ALL NEPSCRIPT GENESIS EXPERIMENTS"
echo "========================================================="
echo "Device: auto-detected"

# 1. RANDOM SEARCH
echo -e "\n>>> Running Random Search..."
python scripts/nas_search.py --strategy random --num-archs 4 --epochs 4 --output-dir "$NAS_ROOT"
echo ">>> Training Random Search Winner..."
python scripts/train_model.py --arch-config "$NAS_ROOT/best_architecture_random.json" --epochs 50

# 2. ADAPTIVE SEARCH
echo -e "\n>>> Running Adaptive Search..."
python scripts/nas_search.py --strategy adaptive --num-archs 4 --epochs 4 --output-dir "$NAS_ROOT"
echo ">>> Training Adaptive Search Winner..."
python scripts/train_model.py --arch-config "$NAS_ROOT/best_architecture_adaptive.json" --epochs 50

# 3. PROGRESSIVE SEARCH
echo -e "\n>>> Running Progressive Search..."
python scripts/nas_search.py --strategy progressive --num-archs 4 --epochs 4 --output-dir "$NAS_ROOT"
echo ">>> Training Progressive Search Winner..."
python scripts/train_model.py --arch-config "$NAS_ROOT/best_architecture_progressive.json" --epochs 50

# 4. MULTI-FIDELITY SEARCH
echo -e "\n>>> Running Multi-Fidelity Search..."
python scripts/nas_search.py --strategy multifidelity --num-archs 4 --output-dir "$NAS_ROOT"
echo ">>> Training Multi-Fidelity Search Winner..."
python scripts/train_model.py --arch-config "$NAS_ROOT/best_architecture_multifidelity.json" --epochs 50

# 5. ADVERSARIAL NAS
echo -e "\n>>> Running Adversarial NAS (Gradient-based)..."
python scripts/nas_search.py --strategy adversarial --epochs 25 --output-dir "$NAS_ROOT"
echo ">>> Training Adversarial NAS Winner..."
python scripts/train_model.py --arch-config "$NAS_ROOT/best_architecture_adversarial.json" --epochs 50

echo -e "\n========================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "========================================================="
